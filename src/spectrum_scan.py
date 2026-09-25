"""
Quét TOÀN PHỔ thay vì chốt cứng chu kỳ 5 phút.

Vì sao cần: `spectral.py` chỉ nhìn tần số 1/5 phút⁻¹. Điều tra nhóm `others` cho thấy có
cả họ chu kỳ 3 phút (gọi đúng 1 lần mỗi 3 phút, 480/1440 điểm khác 0), mà PR cũ hoàn
toàn mù với nó. Nghĩa là mọi con số PR đã báo cáo đều là CẬN DƯỚI.

Ở đây, mỗi chuỗi sai số tự khai báo chu kỳ của nó:
  1. tính periodogram
  2. tìm đỉnh trội (bỏ DC và vùng tần số rất thấp — đó là xu hướng, không phải chu kỳ)
  3. PR_peak = công suất tại đỉnh (±1 bin) / công suất nền trung vị
  4. chu kỳ trội = 1 / tần số đỉnh

Kết quả phụ, và có lẽ là thứ đáng giá nhất: **phân bố chu kỳ trội trên toàn quần thể**.
Nếu workload thật sự bị chi phối bởi lịch hẹn giờ thì phân bố này phải dồn vào đúng các
mốc cron quen thuộc (1, 2, 3, 5, 10, 15, 30, 60 phút) chứ không rải đều.

    env/python.exe src/spectrum_scan.py --day 1
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from pipeline import (RESULTS, SEED, error_series, load_rows, pick_strata,
                      scan_totals)

MIN_PERIOD = 2.0      # Nyquist
MAX_PERIOD = 240.0    # dài hơn 4 giờ thì coi là xu hướng, không phải chu kỳ


def scan(e):
    """
    e: (n_series, T). Trả về (PR tại đỉnh, chu kỳ trội tính bằng phút).
    Chuẩn hoá từng chuỗi để function volume lớn không chi phối.
    """
    x = e - e.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    x = x / sd

    P = np.abs(np.fft.rfft(x, axis=1)) ** 2
    T = e.shape[1]
    freqs = np.fft.rfftfreq(T, d=1.0)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), np.inf)

    band = (periods >= MIN_PERIOD) & (periods <= MAX_PERIOD)
    idx_band = np.flatnonzero(band)

    pr = np.full(e.shape[0], np.nan)
    per = np.full(e.shape[0], np.nan)
    for i in range(e.shape[0]):
        pb = P[i, idx_band]
        if pb.size == 0 or not np.isfinite(pb).any():
            continue
        k = idx_band[int(np.argmax(pb))]
        lo, hi = max(k - 1, 1), k + 2
        peak = P[i, lo:hi].mean()
        mask = np.ones(len(freqs), bool)
        mask[:1] = False            # bỏ DC
        mask[lo:hi] = False         # bỏ chính đỉnh khỏi nền
        bg = np.median(P[i, mask])
        if bg > 0:
            pr[i] = peak / bg
            per[i] = periods[k]
    return pr, per


def period_histogram(per, pr, pr_min=3.0):
    """Gom chu kỳ trội về số nguyên phút gần nhất, chỉ đếm chuỗi có đỉnh đủ rõ."""
    ok = np.isfinite(per) & np.isfinite(pr) & (pr >= pr_min)
    if not ok.any():
        return Counter()
    return Counter(np.round(per[ok]).astype(int).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    meta = scan_totals(args.day)

    # --- 1. hai tầng mẫu: PR tại đỉnh thật, so với PR cũ (chốt cứng 5 phút)
    rows = []
    strata = pick_strata(meta, n=args.n, rng=rng)
    for stratum, idx in strata.items():
        y = load_rows(args.day, idx)
        y_shuf = np.apply_along_axis(rng.permutation, 1, y)
        for fc in ("naive", "seasonal60", "ma10"):
            pr_r, per_r = scan(error_series(y, fc))
            pr_b, _ = scan(error_series(y_shuf, fc))
            ok = np.isfinite(pr_r) & np.isfinite(pr_b)
            rows.append({
                "stratum": stratum, "forecaster": fc, "n": int(ok.sum()),
                "PRpeak_real_median": float(np.median(pr_r[ok])),
                "PRpeak_ctrlB_median": float(np.median(pr_b[ok])),
                "ratio": float(np.median(pr_r[ok]) / np.median(pr_b[ok])),
                "frac_gt10_real": float((pr_r[ok] > 10).mean()),
                "frac_gt10_ctrlB": float((pr_b[ok] > 10).mean()),
                "period_median": float(np.nanmedian(per_r[ok])),
            })
            print(f"[peak] {stratum:7s} {fc:12s} PR real={rows[-1]['PRpeak_real_median']:8.2f} "
                  f"B={rows[-1]['PRpeak_ctrlB_median']:6.2f}  "
                  f"%>10: {rows[-1]['frac_gt10_real']:5.1%} vs {rows[-1]['frac_gt10_ctrlB']:5.1%}")

    # --- 2. phân bố chu kỳ trội trên mẫu lớn, cắt theo trigger
    hist_rows = []
    all_hist = Counter()
    for trig in ("timer", "http", "queue", "orchestration", "event", "storage", "others"):
        pool = np.flatnonzero((meta["Trigger"] == trig).to_numpy()
                              & (meta["total"] >= 100).to_numpy())
        if len(pool) < 30:
            continue
        idx = rng.choice(pool, size=min(400, len(pool)), replace=False)
        y = load_rows(args.day, idx)
        pr, per = scan(error_series(y, "naive"))
        h = period_histogram(per, pr)
        all_hist.update(h)
        tot = sum(h.values())
        top = h.most_common(5)
        hist_rows.append({"trigger": trig, "n": len(idx), "n_with_peak": tot,
                          "frac_with_peak": tot / len(idx),
                          "top_periods": "; ".join(f"{p}p:{c}" for p, c in top)})
        print(f"[hist] {trig:14s} {tot:3d}/{len(idx)} có đỉnh rõ  →  "
              + ", ".join(f"{p} phút ({c})" for p, c in top))

    d = os.path.join(RESULTS, f"d{args.day:02d}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "spectrum_peak.csv"), index=False)
    pd.DataFrame(hist_rows).to_csv(os.path.join(d, "spectrum_periods_by_trigger.csv"), index=False)

    print("\n=== Phân bố chu kỳ trội, gộp mọi trigger (top 15) ===")
    tot = sum(all_hist.values())
    cron = {1, 2, 3, 5, 10, 15, 20, 30, 60, 120}
    hit = 0
    for p, c in all_hist.most_common(15):
        mark = " ← mốc cron quen thuộc" if p in cron else ""
        print(f"  {p:4d} phút: {c:4d}  ({c/tot:5.1%}){mark}")
    hit = sum(c for p, c in all_hist.items() if p in cron)
    print(f"\n  Tổng số chuỗi có đỉnh rõ: {tot}")
    print(f"  Rơi đúng vào mốc cron quen thuộc: {hit} ({hit/tot:.1%})")
    pd.DataFrame(sorted(all_hist.items()), columns=["period_min", "count"]).to_csv(
        os.path.join(d, "spectrum_period_histogram.csv"), index=False)


if __name__ == "__main__":
    main()
