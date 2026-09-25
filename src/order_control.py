"""
Thí nghiệm thứ tự CÓ KIỂM SOÁT — biến quan sát của mục 9 thành cơ chế.

Mục 9 mới chỉ nói: chuỗi thật rẻ hơn hoán vị, và chuỗi càng bền thì càng rẻ. Đó là TƯƠNG QUAN
trên chuỗi thật, mà chuỗi thật khác chuỗi xáo ở nhiều thứ cùng lúc. Ở đây làm ngược lại: giữ
nguyên ĐA TẬP sai số (nên η₁ và η∞ cố định chính xác) và TỰ ĐẶT độ bền của thứ tự.

  Cách dựng: sinh chuỗi ẩn AR(1) với hệ số φ, rồi gán giá trị của đa tập theo thứ hạng của
  chuỗi ẩn (giá trị lớn nhất vào vị trí có giá trị ẩn lớn nhất). φ điều khiển độ bền mà không
  đụng tới đa tập:
      φ = -0.6      xen kẽ cao–thấp (bền âm)
      φ =  0        hoán vị ngẫu nhiên  ← mốc so sánh
      φ =  0.9/0.99 vón cục mạnh
      sorted        đơn điệu tăng (bền tối đa)
  Kèm theo thứ tự THẬT để xem nó rơi vào đâu trên trục đó.

Nếu chi phí giảm đều theo φ thì "độ bền làm rẻ đi" thành phát biểu nhân quả trong mô hình này,
và không còn phụ thuộc vào đặc thù trace Azure, vì các thứ tự này là nhân tạo.

    hd/python.exe src/order_control.py --days 1 2 10
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pipeline import RESULTS, SEED, acf_matrix, tau_int
from verify_c1 import ALGOS, eval_order, pick_series

PHIS = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 0.99]
NREAL = 3          # số lần sinh chuỗi ẩn cho mỗi φ, để trung bình bớt nhiễu của chính chuỗi ẩn
F_COST, THETA = 10.0, 8.0


def ar1_order(d_sorted, phi, rng):
    """Gán đa tập đã sắp xếp lên trục thời gian theo thứ hạng của một chuỗi ẩn AR(1)."""
    n = len(d_sorted)
    u = np.empty(n)
    u[0] = rng.normal()
    s = np.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        u[t] = phi * u[t - 1] + s * rng.normal()
    out = np.empty(n)
    out[np.argsort(u)] = d_sorted
    return out


def persistence(d):
    rho = acf_matrix(d[None, :])[0]
    return float(np.nanmean(rho[2:31])), float(tau_int(rho[None, :])[0])


def worker(job):
    d, R = job["d"], job["R"]
    ds = np.sort(d)
    ss = [SEED, job["day"], job["k"], 7]
    drng = np.random.default_rng(ss + [1])
    XS = [np.asarray(x) for x in job["XS"]] if "XS" in job else None
    from ofl_cost2 import clustered_demand
    XS = [clustered_demand(len(d), drng) for _ in range(R)]

    orders = [("real", d, 0)]
    orng = np.random.default_rng(ss + [2])
    for phi in PHIS:
        for j in range(NREAL):
            orders.append((f"phi{phi:+.2f}", ar1_order(ds, phi, orng), j))
    orders.append(("sorted", ds, 0))

    rows = []
    for name, seq, j in orders:
        am, ti = persistence(seq)
        row = {"day": job["day"], "k": job["k"], "order": name, "rep": j,
               "acf_mean_2_30": am, "tau_int": ti}
        for a, aid in ALGOS.items():
            c, n = eval_order(a, seq, XS, F_COST, THETA, ss + [3, aid])
            row[f"cost_{a}"] = c.mean()
            row[f"nfac_{a}"] = n.mean()
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", default=[1, 2, 10])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    for day in ([] if args.analyze_only else args.days):
        t0 = time.time()
        series = pick_series(day, args.n)
        jobs = [{"day": day, "k": s["k"], "d": s["abs"], "R": args.R} for s in series]
        print(f"[data] d{day:02d}: {len(jobs)} chuỗi ({time.time() - t0:.0f}s)")
        rows = []
        with Pool(args.procs) as pool:
            for i, r in enumerate(pool.imap_unordered(worker, jobs, chunksize=1)):
                rows.extend(r)
                if (i + 1) % 25 == 0:
                    print(f"  ... {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)")
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"order_control_d{day:02d}.csv"), index=False)
        print(f"[save] d{day:02d} ({time.time() - t0:.0f}s)")

    analyze(args.days)


def analyze(days):
    log = []

    def say(s=""):
        print(s); log.append(s)

    df = pd.concat([pd.read_csv(os.path.join(RESULTS, f"order_control_d{d:02d}.csv")) for d in days])
    # mốc: trung bình các lần sinh với φ = 0 (hoán vị ngẫu nhiên) của cùng chuỗi
    base = (df[df.order == "phi+0.00"].groupby(["day", "k"])[[f"cost_{a}" for a in ALGOS]]
            .mean().rename(columns=lambda c: c + "_base"))
    m = df.merge(base, on=["day", "k"])
    for a in ALGOS:
        m[f"rel_{a}"] = 100.0 * (m[f"cost_{a}"] - m[f"cost_{a}_base"]) / m[f"cost_{a}_base"]

    say("=== Chi phí so với thứ tự ngẫu nhiên (%), theo độ bền áp đặt ===")
    say("  cùng đa tập sai số → η₁ và η∞ giống hệt nhau ở mọi dòng")
    say(f"  {'thứ tự':10s} {'acf(2..30)':>10s} {'τ_int':>7s}" +
        "".join(f"{'  ' + a:>11s}" for a in ALGOS))
    order_names = ["phi-0.60", "phi-0.30", "phi+0.00", "phi+0.30", "phi+0.60",
                   "phi+0.90", "phi+0.99", "sorted", "real"]
    for o in order_names:
        g = m[m.order == o]
        if g.empty:
            continue
        say(f"  {o:10s} {g.acf_mean_2_30.median():10.3f} {g.tau_int.median():7.2f}" +
            "".join(f"{g['rel_' + a].median():+11.2f}" for a in ALGOS))

    say("\n=== Kiểm đơn điệu theo φ (chỉ các thứ tự AR(1), bỏ sorted/real) ===")
    from scipy import stats as st  # dùng cho cả kiểm ghép cặp
    sub = m[m.order.str.startswith("phi")].copy()
    sub["phi"] = sub.order.str[3:].astype(float)
    for a in ALGOS:
        for d in days:
            g = sub[sub.day == d]
            rho = st.spearmanr(g.phi, g[f"rel_{a}"]).statistic
            say(f"  {a:6s} d{d:02d}: Spearman(φ, chi phí tương đối) = {rho:+.3f}")

    say("\n=== Kiểm GHÉP CẶP trên từng chuỗi (Spearman gộp ở trên bị nhiễu trong-chuỗi lấn át) ===")
    say("  so trên CÙNG một chuỗi sai số, chỉ khác thứ tự")
    piv = m.pivot_table(index=["day", "k"], columns="order",
                        values=[f"rel_{a}" for a in ALGOS], aggfunc="mean")
    for a in ALGOS:
        for lhs, rhs, lab in [("phi+0.99", "phi+0.00", "φ=0.99 vs ngẫu nhiên"),
                              ("sorted", "phi+0.00", "sắp xếp vs ngẫu nhiên"),
                              ("real", "phi+0.00", "thật vs ngẫu nhiên"),
                              ("real", "phi+0.90", "thật vs φ=0.90 (τ cao hơn thật)")]:
            dif = (piv[(f"rel_{a}", lhs)] - piv[(f"rel_{a}", rhs)]).dropna()
            neg = int((dif < 0).sum())
            p = st.binomtest(neg, len(dif), 0.5).pvalue
            say(f"  {a:6s} {lab:32s} trung vị {dif.median():+7.2f}%  rẻ hơn ở {neg/len(dif):4.0%}  p={p:.1e}")

    say("\n=== Thứ tự THẬT rơi vào đâu ===")
    for a in ALGOS:
        for d in days:
            g = m[(m.order == "real") & (m.day == d)]
            say(f"  {a:6s} d{d:02d}: thật {g['rel_' + a].median():+6.2f}%   "
                f"(mốc ngẫu nhiên = 0 theo định nghĩa)   rẻ hơn ngẫu nhiên ở "
                f"{(g['rel_' + a] < 0).mean():.0%} số chuỗi")

    out = os.path.join(RESULTS, "order_control_summary.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
