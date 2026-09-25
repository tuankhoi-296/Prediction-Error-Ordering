"""
Thí nghiệm quyết định — chốt kết luận âm tính của mục 4e.

Mục 4e kết luận: không thống kê vô hướng nào dự đoán được chênh lệch chi phí ổn định.
Nhưng nó mới dựa trên **2 thuật toán** và **9 ứng viên**, nên kết luận âm tính còn yếu:
có thể tôi chưa thử đúng thống kê, hoặc hai thuật toán kia tình cờ giống nhau ở chỗ nào đó.

Bản này đóng cả hai lỗ hổng:

  BA thuật toán, khác nhau về bản chất:
    threshold  — tin dự đoán tuyệt đối, tất định, có tham số bán kính θ
    meyerson   — tin dự đoán tuyệt đối, NGẪU NHIÊN, không tham số
    trust      — TIN MỘT PHẦN: mở tại (1-ε)·x + ε·p. Đây là nút vặn consistency/robustness
                 kinh điển của Purohit et al. 2018, và là thuật toán duy nhất trong ba cái
                 thực sự "dùng dự đoán một cách có kiểm soát".

  MƯỜI BỐN ứng viên thống kê, trong đó 5 cái mới và đều KHÔNG phụ thuộc ngưỡng tuỳ chọn:
    burst_auc    — diện tích dưới đường burst_mean(q) với q chạy 50..99.
                   Đây là bản xấp xỉ nghèo của burst-tree (Jo et al. 2020): thay vì chọn
                   một ngưỡng p90 tuỳ ý, tích hợp qua MỌI ngưỡng.
    hurst        — số mũ Hurst bằng DFA. Thống kê kinh điển cho phụ thuộc tầm xa.
    spec_slope   — độ dốc log-log của phổ công suất (số mũ 1/f).
    acf_decay    — số lag đầu tiên mà ACF tụt xuống dưới 1/e.
    perm_entropy — entropy hoán vị bậc 3, đo độ phức tạp thứ tự.

Quy tắc phán quyết giữ nguyên, nhưng chặt hơn: một thống kê chỉ "dùng được" nếu tương quan
**cùng dấu, |r| > 0.2, và khoảng tin cậy 95% không chứa 0** trên **CẢ BA thuật toán** và
**CẢ HAI ngày**. Tức 6 điều kiện.

    env/python.exe src/c1_final.py --n-seq 300 --day 1
"""

import argparse
import os

import numpy as np
import pandas as pd

from pipeline import (RESULTS, SEED, acf_matrix, burst_lengths, error_series,
                      load_rows, scan_totals, tau_int)
from ofl_cost2 import clustered_demand, simulate
from robust_c1 import simulate_meyerson
from spectrum_scan import scan


# ───────────────────────────────── thuật toán thứ ba

def simulate_trust(x, p, f_cost, eps, rng):
    """
    Tin một phần: mở facility tại (1-eps)·x_i + eps·p_i, theo luật Meyerson.
    eps = 0 → bỏ qua dự đoán hoàn toàn (robust).  eps = 1 → tin tuyệt đối (consistent).
    Đây là nút vặn của Purohit et al. 2018 đặt vào bối cảnh OFL.
    """
    q = (1.0 - eps) * x + eps * p
    fac = np.empty(len(x)); nfac = 0; serve = 0.0
    for i in range(len(x)):
        if nfac == 0:
            fac[0] = q[i]; nfac = 1
        else:
            d = np.abs(fac[:nfac] - q[i]).min()
            if rng.random() < min(d / f_cost, 1.0):
                fac[nfac] = q[i]; nfac += 1
        serve += np.abs(fac[:nfac] - x[i]).min()
    return f_cost * nfac + serve, nfac


ALGOS = {
    "thr": lambda x, p, rng: simulate(x, p, 10.0, 8.0),
    "mey": lambda x, p, rng: simulate_meyerson(x, p, 10.0, rng),
    "trust": lambda x, p, rng: simulate_trust(x, p, 10.0, 0.5, rng),
}


# ───────────────────────────────── ứng viên thống kê mới

def burst_auc(d, qs=range(50, 100, 5)):
    """Tích hợp độ dài burst trung bình qua MỌI ngưỡng — bỏ được tham số p90 tuỳ ý."""
    v = []
    for q in qs:
        b = burst_lengths(d.reshape(1, -1), q=q)
        v.append(b.mean() if b.size else 0.0)
    return float(np.trapezoid(v, dx=1.0) / max(len(v) - 1, 1))


def hurst_dfa(d, scales=(8, 16, 32, 64, 128, 256)):
    """Số mũ Hurst bằng detrended fluctuation analysis."""
    y = np.cumsum(d - d.mean())
    F = []
    use = []
    for s in scales:
        if len(y) < 4 * s:
            continue
        n = len(y) // s
        seg = y[:n * s].reshape(n, s)
        t = np.arange(s)
        # khử xu hướng tuyến tính từng đoạn
        A = np.vstack([t, np.ones(s)]).T
        coef, *_ = np.linalg.lstsq(A, seg.T, rcond=None)
        resid = seg.T - A @ coef
        F.append(np.sqrt((resid ** 2).mean()))
        use.append(s)
    if len(F) < 3:
        return np.nan
    return float(np.polyfit(np.log(use), np.log(F), 1)[0])


def spec_slope(d):
    """Độ dốc log-log của phổ công suất (số mũ 1/f^beta)."""
    x = d - d.mean()
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x))
    m = (f > 0) & (P > 0)
    if m.sum() < 20:
        return np.nan
    return float(np.polyfit(np.log(f[m]), np.log(P[m]), 1)[0])


def acf_decay(rho):
    """Lag đầu tiên mà ACF tụt dưới 1/e."""
    thr = 1.0 / np.e
    for k in range(1, len(rho)):
        if not np.isfinite(rho[k]) or rho[k] < thr:
            return float(k)
    return float(len(rho))


def perm_entropy(d, order=3):
    """Entropy hoán vị — đo độ phức tạp của THỨ TỰ, bất biến với phép đơn điệu."""
    n = len(d) - order + 1
    if n < 50:
        return np.nan
    idx = np.argsort(np.lib.stride_tricks.sliding_window_view(d, order), axis=1)
    codes = np.ravel_multi_index(idx.T, (order,) * order)
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    import math
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(order)))


def all_stats(d):
    x = d.reshape(1, -1)
    rho = acf_matrix(x, max_lag=60)[0]
    pr, per = scan(x)
    b = burst_lengths(x, q=90)
    m = d.mean()
    lags = np.arange(2, 61)
    return {
        # cũ
        "tau_int": float(tau_int(rho.reshape(1, -1))[0]) if np.isfinite(rho[1]) else np.nan,
        "PR_peak": float(pr[0]), "period": float(per[0]),
        "acf_mean_2_30": float(np.nanmean(rho[2:31])),
        "acf_max_2_60": float(np.nanmax(rho[2:61])),
        "periodic5": float(np.nanmean(rho[lags][lags % 5 == 0])
                           - np.nanmean(rho[lags][lags % 5 != 0])),
        "burst_max": float(b.max()) if b.size else 0.0,
        "burst_mean": float(b.mean()) if b.size else 0.0,
        "CV": float(d.std() / m) if m > 0 else np.nan,
        # mới, không phụ thuộc ngưỡng
        "burst_auc": burst_auc(d),
        "hurst": hurst_dfa(d),
        "spec_slope": spec_slope(d),
        "acf_decay": acf_decay(rho),
        "perm_entropy": perm_entropy(d),
    }


STATS = ["tau_int", "PR_peak", "period", "acf_mean_2_30", "acf_max_2_60", "periodic5",
         "burst_max", "burst_mean", "CV", "burst_auc", "hurst", "spec_slope",
         "acf_decay", "perm_entropy"]
LOGS = {"tau_int", "PR_peak", "period", "burst_auc", "acf_decay"}


def gaps(d, rng, REP=20):
    out = {}
    for nm, fn in ALGOS.items():
        a = b = c = 0.0
        for _ in range(REP):
            x = clustered_demand(len(d), rng)
            a += fn(x, x + d, rng)[0]
            b += fn(x, x + rng.permutation(d), rng)[0]
            c += fn(x, x + rng.permutation(d), rng)[0]
        a, b, c = a / REP, b / REP, c / REP
        out[f"gap_{nm}"] = 100.0 * (a - b) / b
        out[f"noise_{nm}"] = 100.0 * abs(b - c) / c
        out[f"snr_{nm}"] = abs(out[f"gap_{nm}"]) / out[f"noise_{nm}"] if out[f"noise_{nm}"] > 0 else np.inf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=300)
    ap.add_argument("--day", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    meta = scan_totals(args.day)
    pool = np.flatnonzero((meta["total"] >= 500).to_numpy())
    idx = rng.choice(pool, size=min(args.n_seq, len(pool)), replace=False)
    trig = meta["Trigger"].to_numpy()[idx]
    y = load_rows(args.day, idx)
    print(f"[data] d{args.day:02d}: {len(idx)} chuỗi, 3 thuật toán, {len(STATS)} ứng viên")

    rows = []
    for k in range(len(idx)):
        fc = ("naive", "seasonal60", "ma10")[k % 3]
        e = error_series(y[k:k + 1], fc)[0]
        if e.std() == 0 or e.mean() == 0:
            continue
        d = e / e.mean()
        s = all_stats(d)
        if not np.isfinite(s["tau_int"]):
            continue
        s.update(gaps(d, rng))
        s.update({"trigger": trig[k], "forecaster": fc})
        rows.append(s)
        if len(rows) % 50 == 0:
            print(f"  ... {len(rows)}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, f"c1_final_d{args.day:02d}.csv")
    df.to_csv(out, index=False)
    print(f"\n[save] {out}   n = {len(df)}")

    rel = df[(df.snr_thr >= 3) & (df.snr_mey >= 3) & (df.snr_trust >= 3)]
    print(f"\nSố chuỗi vượt mốc nhiễu ở CẢ BA thuật toán: {len(rel)}/{len(df)}")

    def table(sub, name):
        print(f"\n=== {name} (n = {len(sub)}) ===")
        print(f"{'thống kê':15s}" + "".join(f"{'r('+a+')':>22s}" for a in ALGOS) + "   3/3?")
        winners = []
        for st in STATS:
            v = sub[st].to_numpy(float)
            v = np.log10(np.clip(v, 1e-9, None)) if st in LOGS else v
            ok = np.isfinite(v)
            if ok.sum() < 10:
                continue
            line = f"{st:15s}"; rs = []
            for a in ALGOS:
                g = sub[f"gap_{a}"].to_numpy(float)[ok]
                r = np.corrcoef(v[ok], g)[0, 1]
                n = ok.sum(); z = np.arctanh(np.clip(r, -.999, .999)); se = 1 / np.sqrt(n - 3)
                lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
                rs.append((r, lo, hi))
                line += f" {r:+7.3f}[{lo:+5.2f},{hi:+5.2f}]"
            good = all(abs(r) > 0.2 and lo * hi > 0 for r, lo, hi in rs) and \
                   len({np.sign(r) for r, _, _ in rs}) == 1
            if good:
                winners.append(st)
            print(line + ("   ✅" if good else "   —"))
        return winners

    w1 = table(df, "TẤT CẢ")
    w2 = table(rel, "CHỈ chuỗi vượt nhiễu ở cả ba thuật toán") if len(rel) >= 15 else []
    print(f"\n=== ỨNG VIÊN QUA ĐƯỢC CẢ BA THUẬT TOÁN ===")
    print(f"  toàn bộ mẫu : {w1 if w1 else 'KHÔNG CÓ'}")
    print(f"  mẫu vượt nhiễu: {w2 if w2 else 'KHÔNG CÓ'}")

    print("\n=== Độ lớn chênh lệch chi phí theo thuật toán ===")
    for a in ALGOS:
        g = df[f"gap_{a}"]
        print(f"  {a:6s} trung vị |{g.abs().median():6.2f}|%  p90 |{g.abs().quantile(.9):7.2f}|%  "
              f"max |{g.abs().max():7.2f}|%  vượt nhiễu {(df['snr_'+a]>=3).mean():.0%}")

    print("\n=== Tương quan chi phí GIỮA các thuật toán (cùng chuỗi) ===")
    A = list(ALGOS)
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            print(f"  {A[i]} vs {A[j]}: {np.corrcoef(df['gap_'+A[i]], df['gap_'+A[j]])[0,1]:+.3f}")


if __name__ == "__main__":
    main()
