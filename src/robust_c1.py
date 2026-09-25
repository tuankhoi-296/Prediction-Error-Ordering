"""
Kiểm độ bền của kết luận C1 theo hai chiều độc lập.

Kết luận cần kiểm (từ `which_stat.py`, mới chỉ chạy trên d01 + một thuật toán):
    PR (trục chu kỳ)  tương quan +0.74 với chênh lệch chi phí có dấu
    τ_int (trục vón cục) tương quan +0.03

Hai chiều kiểm:
  --day 2 / 10   : ngày khác  → loại khả năng d01 là ngày cá biệt
  --algo meyerson: thuật toán khác → loại khả năng kết luận là hiện vật của luật ngưỡng

Meyerson (1999) là thuật toán OFL kinh điển: khi request đến cách facility gần nhất một
khoảng d, mở facility mới với xác suất min(d/f, 1). Bản tăng cường dự đoán ở đây tính d
và đặt facility theo VỊ TRÍ DỰ ĐOÁN, còn chi phí phục vụ vẫn tính theo vị trí thật.
Khác hẳn luật ngưỡng ở chỗ nó **ngẫu nhiên** và **không có tham số θ**.

    env/python.exe src/robust_c1.py --day 1 --algo meyerson
    env/python.exe src/robust_c1.py --day 2 --algo threshold
"""

import argparse
import os

import numpy as np
import pandas as pd

from pipeline import (RESULTS, SEED, acf_matrix, error_series, load_rows,
                      pick_strata, scan_totals, tau_int)
from ofl_cost2 import clustered_demand, simulate
from spectrum_scan import scan


def simulate_meyerson(x, p, f_cost, rng):
    """Meyerson có dự đoán: mở tại p_i với xác suất min(dist(p_i, F)/f, 1)."""
    fac = np.empty(len(x)); nfac = 0; serve = 0.0
    for i in range(len(x)):
        if nfac == 0:
            fac[0] = p[i]; nfac = 1
        else:
            d = np.abs(fac[:nfac] - p[i]).min()
            if rng.random() < min(d / f_cost, 1.0):
                fac[nfac] = p[i]; nfac += 1
        serve += np.abs(fac[:nfac] - x[i]).min()
    return f_cost * nfac + serve, nfac


def cost_gap(d, rng, algo, f_cost=10.0, theta=8.0, REP=30):
    run = ((lambda x, p: simulate(x, p, f_cost, theta)) if algo == "threshold"
           else (lambda x, p: simulate_meyerson(x, p, f_cost, rng)))
    real = p1s = p2s = 0.0
    for _ in range(REP):
        x = clustered_demand(len(d), rng)
        real += run(x, x + d)[0]
        p1, p2 = rng.permutation(d), rng.permutation(d)
        p1s += run(x, x + p1)[0]
        p2s += run(x, x + p2)[0]
    real, p1s, p2s = real / REP, p1s / REP, p2s / REP
    return 100.0 * (real - p1s) / p1s, 100.0 * abs(p1s - p2s) / p2s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--algo", choices=["threshold", "meyerson"], default="threshold")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    N = 2500
    meta = scan_totals(args.day)
    strata = pick_strata(meta, n=200, rng=np.random.default_rng(SEED))
    sources = dict(strata)
    for trig in ("timer", "http", "queue", "orchestration", "event", "storage", "others"):
        pool = np.flatnonzero((meta["Trigger"] == trig).to_numpy()
                              & (meta["total"] >= 100).to_numpy())
        if len(pool) >= 30:
            sources[trig] = np.random.default_rng(SEED).choice(
                pool, size=min(200, len(pool)), replace=False)

    rows = []
    for src, idx in sources.items():
        y = load_rows(args.day, idx)
        for fc in ("naive", "seasonal60", "ma10"):
            e = error_series(y, fc)
            e = e[e.std(axis=1) > 0]
            if e.shape[0] == 0:
                continue
            order = np.argsort(-e.std(axis=1))
            d0 = np.concatenate([e[i] for i in order[:N // e.shape[1] + 1]])[:N]
            if d0.std() == 0 or d0.mean() == 0:
                continue
            d = d0 / d0.mean()
            rho = acf_matrix(d.reshape(1, -1), max_lag=60)
            t = float(tau_int(rho)[0]) if np.isfinite(rho[0, 1]) else np.nan
            pr, per = scan(d.reshape(1, -1))
            sig, noise = cost_gap(d, rng, args.algo)
            rows.append({"source": src, "forecaster": fc, "tau_int": t,
                         "PR_peak": float(pr[0]), "period": float(per[0]),
                         "signal_pct": sig, "noise_pct": noise,
                         "snr": abs(sig) / noise if noise > 0 else np.inf})
            print(f"[{args.algo[:4]}] d{args.day:02d} {src:14s} {fc:11s} "
                  f"τ={t:7.2f} PR={float(pr[0]):9.1f}  →  {sig:+7.2f}% (nhiễu {noise:4.2f}%)")

    df = pd.DataFrame(rows)
    tag = f"d{args.day:02d}_{args.algo}"
    out = os.path.join(RESULTS, f"robust_c1_{tag}.csv")
    df.to_csv(out, index=False)

    ok = df[np.isfinite(df.tau_int) & np.isfinite(df.PR_peak)]
    rel = ok[ok.snr >= 3]
    print(f"\n=== {tag} — tương quan với chênh lệch CÓ DẤU ===")
    print(f"  n = {len(ok)} chuỗi, trong đó {len(rel)} vượt mốc nhiễu (SNR ≥ 3)")
    for nm, sub in (("tất cả", ok), ("SNR≥3", rel)):
        if len(sub) < 4:
            continue
        lp = np.log10(sub.PR_peak.to_numpy(float))
        lt = np.log10(sub.tau_int.to_numpy(float))
        s = sub.signal_pct.to_numpy(float)
        print(f"  [{nm:6s}] log(PR) r = {np.corrcoef(lp, s)[0,1]:+.3f}   "
              f"log(τ_int) r = {np.corrcoef(lt, s)[0,1]:+.3f}")
    print(f"  Chênh lệch lớn nhất: {df.signal_pct.abs().max():.1f}%  "
          f"| nhiễu trung vị: {df.noise_pct.median():.2f}%")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
