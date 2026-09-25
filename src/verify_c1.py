"""
Kiểm chứng lại C1 — đọc lại c1_final_d0*.csv (2026-09-15) lộ ra ba chỗ KETQUA-D1 chưa kiểm.

  1. CHIỀU. Chênh chi phí có dấu lệch hẳn về âm (chuỗi thật RẺ hơn hoán vị ở ~2/3 số chuỗi),
     nhưng KETQUA chỉ ghi "có cả hai dấu". Nếu thật thì câu chuyện "sai cùng nhau thì hại"
     bị đảo chiều.
  2. MỐC NHIỄU. c1_final ước lượng nhiễu từ MỘT cặp hoán vị, nên SNR = |X|/|Y| với X, Y cùng
     phân phối → dưới null vẫn có ~20% chuỗi đạt SNR ≥ 3. Thay bằng phép kiểm hoán vị đúng
     nghĩa: M hoán vị chạy trên CÙNG R thể hiện nhu cầu và cùng số ngẫu nhiên của thuật toán
     (common random numbers). Thêm một hoán vị "giả làm chuỗi thật" để tự đo tỉ lệ báo
     động giả của chính phép kiểm.
  3. ĐỘ TIN CẬY. Chia đôi R thể hiện → tương quan split-half của gap qua các chuỗi. Đó là
     trần cho mọi tương quan "thống kê ↔ chi phí"; kết luận âm tính chỉ có nghĩa nếu trần cao.

  Thêm hai việc:
  4. CƠ CHẾ. Tách chi phí = f·(số facility) + (chi phí phục vụ).
  5. ĐỘ VỮNG CỦA CHIỀU theo lựa chọn mô hình: sai số có dấu thay vì |e|, độ cục bộ nhu cầu
     (dwell 5 / 200), giá facility f = 2 / 50.

    hd/python.exe src/verify_c1.py --days 1 2 10
    hd/python.exe src/verify_c1.py --analyze-only
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from pipeline import (RESULTS, SEED, acf_matrix, forecast, load_rows, scan_totals, tau_int)
from ofl_cost2 import clustered_demand, simulate
from robust_c1 import simulate_meyerson
from c1_final import LOGS, STATS, all_stats, simulate_trust

ALGOS = {"thr": 0, "mey": 1, "trust": 2}
VARIANTS = {
    # tên       vid  sai số   dwell   f     θ
    "base":    (0, "abs",    40,  10.0, 8.0),
    "signed":  (1, "signed", 40,  10.0, 8.0),
    "dwell5":  (2, "abs",     5,  10.0, 8.0),
    "dwell200": (3, "abs",  200,  10.0, 8.0),
    "f2":      (4, "abs",    40,   2.0, 8.0),
    "f50":     (5, "abs",    40,  50.0, 8.0),
}


# ───────────────────────────────── mô phỏng

def run_algo(a, x, p, f, theta, seed):
    rng = np.random.default_rng(seed)
    if a == "thr":
        return simulate(x, p, f, theta)
    if a == "mey":
        return simulate_meyerson(x, p, f, rng)
    return simulate_trust(x, p, f, 0.5, rng)


def eval_order(a, d, XS, f, theta, base):
    """Chi phí của một thứ tự sai số trên R thể hiện nhu cầu; số ngẫu nhiên cố định theo (base, r)."""
    c = np.empty(len(XS)); n = np.empty(len(XS))
    for r, x in enumerate(XS):
        c[r], n[r] = run_algo(a, x, x + d, f, theta, base + [r])
    return c, n


def worker(job):
    d, M, R = job["d"], job["M"], job["R"]
    vid, _, dwell, f, theta = VARIANTS[job["variant"]]
    ss = [SEED, job["day"], job["k"], vid]
    drng = np.random.default_rng(ss + [1])
    XS = [clustered_demand(len(d), drng, dwell=dwell) for _ in range(R)]
    prng = np.random.default_rng(ss + [2])
    perms = [prng.permutation(d) for _ in range(M + 1)]     # perms[M] đóng vai "chuỗi thật giả"
    h = R // 2
    rows = []
    for a, aid in ALGOS.items():
        base = ss + [3, aid]
        cr, nr = eval_order(a, d, XS, f, theta, base)
        CP = np.empty((M, R)); NP = np.empty((M, R))
        for m in range(M):
            CP[m], NP[m] = eval_order(a, perms[m], XS, f, theta, base)
        cs, _ = eval_order(a, perms[M], XS, f, theta, base)
        pm = CP.mean(1); mu, sd = pm.mean(), pm.std(ddof=1)
        real, sham = cr.mean(), cs.mean()
        rows.append({
            "day": job["day"], "k": job["k"], "variant": job["variant"], "algo": a,
            "forecaster": job["forecaster"], "trigger": job["trigger"],
            "cost_real": real, "perm_mean": mu, "perm_sd": sd,
            "perm_min": pm.min(), "perm_max": pm.max(),
            "gap": 100.0 * (real - mu) / mu, "z": (real - mu) / sd if sd > 0 else 0.0,
            "p_lo": (1 + (pm <= real).sum()) / (M + 1), "p_hi": (1 + (pm >= real).sum()) / (M + 1),
            "sham_gap": 100.0 * (sham - mu) / mu,
            "sham_p_lo": (1 + (pm <= sham).sum()) / (M + 1),
            "sham_p_hi": (1 + (pm >= sham).sum()) / (M + 1),
            "gap_half_a": 100.0 * (cr[:h].mean() - CP[:, :h].mean()) / CP[:, :h].mean(),
            "gap_half_b": 100.0 * (cr[h:].mean() - CP[:, h:].mean()) / CP[:, h:].mean(),
            "nfac_real": nr.mean(), "nfac_perm": NP.mean(),
            "serve_real": real - f * nr.mean(), "serve_perm": mu - f * NP.mean(),
        })
    return rows


# ───────────────────────────────── chọn chuỗi

def gini_md(d):
    """Kỳ vọng |d_i − d_j| với i ≠ j — tức E|Δd| nếu thứ tự hoàn toàn ngẫu nhiên."""
    s = np.sort(d); n = len(s)
    return float(2.0 * np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / (n * (n - 1)))


def pick_series(day, n_need):
    rng = np.random.default_rng(SEED)
    meta = scan_totals(day)
    pool = np.flatnonzero((meta["total"] >= 500).to_numpy())
    idx = rng.choice(pool, size=min(3 * n_need, len(pool)), replace=False)
    trig = meta["Trigger"].to_numpy()[idx]
    y = load_rows(day, idx)
    out = []
    for k in range(len(idx)):
        fc = ("naive", "seasonal60", "ma10")[k % 3]
        yhat, w = forecast(y[k:k + 1], fc)
        s = (yhat - y[k:k + 1])[0, w:].astype(float)
        e = np.abs(s)
        if e.std() == 0 or e.mean() == 0:
            continue
        d_abs = e / e.mean()
        st = all_stats(d_abs)
        if not np.isfinite(st["tau_int"]):
            continue
        rho = acf_matrix(d_abs[None, :])[0]
        st.update({"acf1": float(rho[1]),
                   "smooth": float(np.abs(np.diff(d_abs)).mean() / gini_md(d_abs))})
        out.append({"k": k, "forecaster": fc, "trigger": trig[k],
                    "abs": d_abs, "signed": s / e.mean(), "stats": st})
        if len(out) == n_need:
            break
    return out


def run_day(day, n_base, n_var, M, R, M_var, R_var, procs):
    t0 = time.time()
    series = pick_series(day, n_base)
    print(f"[data] d{day:02d}: {len(series)} chuỗi ({time.time() - t0:.0f}s)")
    jobs = []
    for i, s in enumerate(series):
        for v in VARIANTS:
            if v != "base" and i >= n_var:
                continue
            err = VARIANTS[v][1]
            jobs.append({"day": day, "k": s["k"], "forecaster": s["forecaster"],
                         "trigger": s["trigger"], "variant": v, "d": s[err],
                         "M": M if v == "base" else M_var, "R": R if v == "base" else R_var})
    rows = []
    with Pool(procs) as pool:
        for j, r in enumerate(pool.imap_unordered(worker, jobs, chunksize=1)):
            rows.extend(r)
            if (j + 1) % 100 == 0:
                print(f"  ... {j + 1}/{len(jobs)} việc ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    st = pd.DataFrame([{"day": day, "k": s["k"], **s["stats"]} for s in series])
    df.to_csv(os.path.join(RESULTS, f"verify_c1_d{day:02d}.csv"), index=False)
    st.to_csv(os.path.join(RESULTS, f"verify_c1_stats_d{day:02d}.csv"), index=False)
    print(f"[save] d{day:02d}: {len(df)} dòng ({time.time() - t0:.0f}s)")


# ───────────────────────────────── phân tích

def bh_reject(p, q=0.05):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    k = np.flatnonzero(ok).max() + 1 if ok.any() else 0
    rej = np.zeros(n, bool); rej[o[:k]] = True
    return rej


def analyze(days, log):
    def say(s=""):
        print(s); log.append(s)

    df = pd.concat([pd.read_csv(os.path.join(RESULTS, f"verify_c1_d{d:02d}.csv")) for d in days])
    st = pd.concat([pd.read_csv(os.path.join(RESULTS, f"verify_c1_stats_d{d:02d}.csv")) for d in days])
    df["p2"] = np.minimum(1.0, 2 * np.minimum(df.p_lo, df.p_hi))
    df["sham_p2"] = np.minimum(1.0, 2 * np.minimum(df.sham_p_lo, df.sham_p_hi))
    M = int(round(1 / df.p_lo.min())) - 1
    say(f"Số hoán vị mỗi phép kiểm (base): M = {M}  → p nhỏ nhất đạt được = {1/(M+1):.3f}")

    say("\n=== 1+2. CHIỀU và Ý NGHĨA — phép kiểm hoán vị, cùng thể hiện nhu cầu ===")
    say("  thật<hv: tỉ lệ chuỗi thật rẻ hơn trung bình hoán vị | sign-p: kiểm dấu | "
        "rẻ*/đắt*: tỉ lệ p một phía < 0.025 | FDR: số chuỗi qua BH q=0.05 | "
        "giả: tỉ lệ báo động giả của 'chuỗi giả' (kỳ vọng ≈ 5%)")
    hdr = (f"  {'biến thể':9s} {'thuật toán':6s} {'ngày':>4s} {'n':>4s} {'gap trung vị':>12s} "
           f"{'thật<hv':>8s} {'sign-p':>8s} {'rẻ*':>6s} {'đắt*':>6s} {'FDR':>9s} {'giả':>6s}")
    say(hdr)
    for v in VARIANTS:
        for a in ALGOS:
            for d in days:
                g = df[(df.variant == v) & (df.algo == a) & (df.day == d)]
                if g.empty:
                    continue
                n = len(g); neg = int((g.gap < 0).sum())
                sp = stats.binomtest(neg, n, 0.5).pvalue
                rej = bh_reject(g.p2.to_numpy())
                say(f"  {v:9s} {a:6s} {d:4d} {n:4d} {g.gap.median():+11.2f}% "
                    f"{neg / n:8.0%} {sp:8.1e} {(g.p_lo < .025).mean():6.0%} "
                    f"{(g.p_hi < .025).mean():6.0%} {rej.sum():4d}/{n:<4d} {(g.sham_p2 < .05).mean():6.0%}")

    b = df[df.variant == "base"]
    say("\n=== 2b. So với tiêu chí SNR ≥ 3 cũ — cùng dữ liệu, tiêu chí đã hiệu chuẩn ===")
    for a in ALGOS:
        g = b[b.algo == a]
        say(f"  {a:6s} p hai phía < 0.05: {(g.p2 < .05).mean():5.0%}   "
            f"|z| trung vị {g.z.abs().median():.2f}   chuỗi giả p<0.05: {(g.sham_p2 < .05).mean():4.0%}")

    say("\n=== 3. ĐỘ TIN CẬY của gap (split-half, Spearman–Brown) — trần cho mọi tương quan ===")
    for a in ALGOS:
        for d in days:
            g = b[(b.algo == a) & (b.day == d)]
            r = np.corrcoef(g.gap_half_a, g.gap_half_b)[0, 1]
            say(f"  {a:6s} d{d:02d}: r_half = {r:+.3f}   độ tin cậy = {2 * r / (1 + r):+.3f}")

    say("\n=== 4. CƠ CHẾ — chi phí = f·nfac + phục vụ (base) ===")
    for a in ALGOS:
        g = b[b.algo == a]
        dn = 100 * (g.nfac_real - g.nfac_perm) / g.nfac_perm
        ds = 100 * (g.serve_real - g.serve_perm) / g.serve_perm
        f = VARIANTS["base"][3]
        share = (f * (g.nfac_real - g.nfac_perm)) / (g.cost_real - g.perm_mean)
        say(f"  {a:6s} Δnfac trung vị {dn.median():+6.2f}%  Δphục vụ trung vị {ds.median():+6.2f}%  "
            f"nfac_thật<nfac_hv ở {(dn < 0).mean():4.0%}  "
            f"phần chênh do facility (trung vị) {share.replace([np.inf, -np.inf], np.nan).median():+.2f}")

    say("\n=== 5. ĐỘ TRẢI chi phí giữa các hoán vị (kiểm lại mục 4h, có CRN) ===")
    for a in ALGOS:
        g = b[b.algo == a]
        say(f"  {a:6s} CV trung vị {100 * (g.perm_sd / g.perm_mean).median():.2f}%   "
            f"(max−min)/mean trung vị {100 * ((g.perm_max - g.perm_min) / g.perm_mean).median():.2f}%")

    say("\n=== 6. THỐNG KÊ NÀO dự đoán z? (Spearman, base; ✅ = |ρ|>0.2, cùng dấu, mọi thuật toán × mọi ngày) ===")
    cand = list(STATS) + ["acf1", "smooth"]
    m = b.merge(st, on=["day", "k"])
    say(f"  {'thống kê':14s}" + "".join(f"{a + '/d' + str(d):>11s}" for a in ALGOS for d in days))
    for s_ in cand:
        vals = []
        line = f"  {s_:14s}"
        for a in ALGOS:
            for d in days:
                g = m[(m.algo == a) & (m.day == d)]
                x = g[s_].to_numpy(float)
                ok = np.isfinite(x)
                rho = stats.spearmanr(x[ok], g.z.to_numpy()[ok]).statistic if ok.sum() > 10 else np.nan
                vals.append(rho); line += f"{rho:+11.3f}"
        vals = np.array(vals)
        good = np.all(np.abs(vals) > 0.2) and len(set(np.sign(vals))) == 1
        say(line + ("  ✅" if good else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-base", type=int, default=300)
    ap.add_argument("--n-var", type=int, default=80)
    ap.add_argument("--M", type=int, default=60)
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--M-var", type=int, default=40)
    ap.add_argument("--R-var", type=int, default=8)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    if not args.analyze_only:
        for d in args.days:
            run_day(d, args.n_base, args.n_var, args.M, args.R, args.M_var, args.R_var, args.procs)
    log = []
    analyze(args.days, log)
    out = os.path.join(RESULTS, "verify_c1_summary.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
