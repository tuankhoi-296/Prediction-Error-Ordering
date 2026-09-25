"""
Thí nghiệm C1, bản 2 — sửa hai lỗi thiết kế của bản 1.

Bản 1 (`ofl_cost.py`) cho kết quả ±1–8%, dấu đảo lung tung. Chẩn đoán:

  LỖI 1 — request đi random walk KHÔNG GIỚI HẠN.
    Walk trôi đi mãi, nên tại mọi thời điểm gần như chẳng có facility cũ nào còn hữu ích.
    Mà cơ chế khiến THỨ TỰ sai số có ý nghĩa lại chính là: "lúc dự đoán sai nặng, quanh
    đây đã có sẵn facility tốt chưa?" Với walk không giới hạn thì câu trả lời luôn là
    "chưa", nên thứ tự không thể ảnh hưởng gì. Bản 1 đã thiết kế mất chính hiệu ứng
    cần đo.
    → Sửa: nhu cầu tập trung ở K CỤM CỐ ĐỊNH (đúng thực tế edge server: dân cư ở
      những thành phố cố định), request chuyển cụm chậm để giữ tính cục bộ thời gian.

  LỖI 2 — không có mốc nhiễu.
    Không biết 5% là tín hiệu hay dao động. Bản 1 vô dụng ở chỗ này.
    → Sửa: thêm control NULL — so HAI hoán vị độc lập với nhau. Cả hai đều có cùng η và
      đều không có cấu trúc thời gian, nên mọi chênh lệch giữa chúng LÀ nhiễu.
      Kết luận chỉ hợp lệ nếu |thật − hoán vị| vượt hẳn |hoán vị₁ − hoán vị₂|.
      Đây đúng là vai trò mà Control B đã đóng cho phần đo ACF.

    env/python.exe src/ofl_cost2.py
"""

import os

import numpy as np
import pandas as pd

from pipeline import RESULTS, SEED, error_series, load_rows, pick_strata, scan_totals


def simulate(x, p, f_cost, theta):
    """Mở facility dựa trên p (cái nhìn thấy), trả chi phí phục vụ theo x (cái xảy ra thật)."""
    fac = np.empty(len(x)); nfac = 0; serve = 0.0
    for i in range(len(x)):
        if nfac == 0:
            fac[0] = p[i]; nfac = 1
        elif np.abs(fac[:nfac] - p[i]).min() > theta:
            fac[nfac] = p[i]; nfac += 1
        serve += np.abs(fac[:nfac] - x[i]).min()
    return f_cost * nfac + serve, nfac


def clustered_demand(n, rng, K=6, spread=60.0, sigma=1.0, dwell=40):
    """
    K cụm cố định trên đoạn thẳng. Request ở lại một cụm trung bình `dwell` bước rồi
    nhảy sang cụm khác — giữ tính cục bộ thời gian mà không trôi đi vô hạn.
    """
    centers = np.linspace(0, spread, K)
    out = np.empty(n)
    c = rng.integers(K)
    for i in range(n):
        if rng.random() < 1.0 / dwell:
            c = rng.integers(K)
        out[i] = centers[c] + rng.normal(0, sigma)
    return out


def get_error_sequence(day, stratum, forecaster, n_need):
    rng = np.random.default_rng(SEED)
    meta = scan_totals(day)
    idx = pick_strata(meta, n=200, rng=rng)[stratum]
    e = error_series(load_rows(day, idx), forecaster)
    e = e[e.std(axis=1) > 0]
    order = np.argsort(-e.std(axis=1))
    return np.concatenate([e[i] for i in order[:n_need // e.shape[1] + 1]])[:n_need]


def main():
    rng = np.random.default_rng(SEED)
    N, REP = 2500, 40
    seqs = {
        "top/naive": get_error_sequence(1, "top", "naive", N),
        "top/seasonal60": get_error_sequence(1, "top", "seasonal60", N),
        "random/naive": get_error_sequence(1, "random", "naive", N),
    }

    rows = []
    for name, d0 in seqs.items():
        d = d0 / (d0.mean() + 1e-12)     # thang: sai số trung bình = 1 đơn vị khoảng cách
        for f_cost in (2.0, 10.0, 50.0):
            for theta in (1.0, 3.0, 8.0):
                acc = {"real": 0.0, "p1": 0.0, "p2": 0.0,
                       "n_real": 0.0, "n_p1": 0.0}
                for _ in range(REP):
                    x = clustered_demand(N, rng)
                    p1 = rng.permutation(d)
                    p2 = rng.permutation(d)
                    cr, nr = simulate(x, x + d, f_cost, theta)
                    c1, n1 = simulate(x, x + p1, f_cost, theta)
                    c2, _ = simulate(x, x + p2, f_cost, theta)
                    acc["real"] += cr; acc["p1"] += c1; acc["p2"] += c2
                    acc["n_real"] += nr; acc["n_p1"] += n1
                for k in acc: acc[k] /= REP
                signal = 100.0 * (acc["real"] - acc["p1"]) / acc["p1"]
                noise = 100.0 * abs(acc["p1"] - acc["p2"]) / acc["p2"]
                rows.append({
                    "seq": name, "f": f_cost, "theta": theta,
                    "cost_real": acc["real"], "cost_perm": acc["p1"], "cost_perm2": acc["p2"],
                    "signal_pct": signal, "noise_pct": noise,
                    "snr": abs(signal) / noise if noise > 0 else np.inf,
                    "nfac_real": acc["n_real"], "nfac_perm": acc["n_p1"],
                })
                print(f"[run ] {name:16s} f={f_cost:5.1f} θ={theta:4.1f}  "
                      f"tín hiệu={signal:+7.2f}%  nhiễu={noise:5.2f}%  "
                      f"SNR={rows[-1]['snr']:5.1f}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "ofl_cost_c1_v2.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)

    print("\n=== Tín hiệu (thật vs hoán vị) so với nhiễu (hoán vị vs hoán vị), % ===")
    print(df.pivot_table(index=["seq", "theta"], columns="f",
                         values="signal_pct").to_string(float_format=lambda v: f"{v:+7.2f}"))
    print("\n--- mốc nhiễu ---")
    print(df.pivot_table(index=["seq", "theta"], columns="f",
                         values="noise_pct").to_string(float_format=lambda v: f"{v:6.2f}"))

    print("\n=== PHÁN QUYẾT ===")
    strong = df[(df.snr >= 3) & (df.signal_pct.abs() >= 2)]
    print(f"  Cấu hình có |tín hiệu| ≥ 2% VÀ SNR ≥ 3: {len(strong)}/{len(df)}")
    if len(strong):
        print(strong[["seq", "f", "theta", "signal_pct", "noise_pct", "snr",
                      "nfac_real", "nfac_perm"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\n  Tín hiệu lớn nhất: {df.signal_pct.abs().max():.2f}%  "
          f"| nhiễu trung vị: {df.noise_pct.median():.2f}%  "
          f"| SNR lớn nhất: {df.snr.max():.1f}")
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
