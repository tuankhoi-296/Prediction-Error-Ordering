"""
Hình H, I, J cho mục 9 của KETQUA-D1.

  H — phân bố chênh chi phí (thật vs hoán vị) trên 900 chuỗi × 3 thuật toán × 3 ngày,
      đặt cạnh phân bố của "chuỗi giả" (một hoán vị đóng vai chuỗi thật) làm null.
  I — độ bền của sai số vs z, thuật toán threshold, ba ngày chồng lên nhau.
  J — chi phí theo độ bền ÁP ĐẶT (thí nghiệm thứ tự có kiểm soát): cùng đa tập, đổi thứ tự.
  K — phản ví dụ TẤT ĐỊNH: cùng đa tập (η₁, η∞ bằng nhau chính xác), thứ tự dựng sẵn,
      chi phí lệch tới 2.4 lần. Kèm phép đảo thời gian cho thấy lợi thế của thứ tự thật
      KHÔNG đến từ vị trí sớm/muộn.

    hd/python.exe src/fig_verify.py
"""
import os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DAYS = [1, 2, 10]
ALG = ["thr", "mey", "trust"]
ALGN = {"thr": "threshold (deterministic)", "mey": "Meyerson (randomised)", "trust": "trust ε=0.5"}
C = {"thr": "#c1121f", "mey": "#1d3557", "trust": "#2a9d8f"}


def load(prefix):
    return pd.concat([pd.read_csv(os.path.join(R, f"{prefix}_d{d:02d}.csv")) for d in DAYS])


# ────────────────────────────────── Hình H
b = load("verify_c1")
b = b[b.variant == "base"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
rows = []
for ax, a in zip(axes, ALG):
    g = b[b.algo == a]
    lim = np.quantile(np.abs(g.gap), .97)
    bins = np.linspace(-lim, lim, 41)
    ax.hist(np.clip(g.sham_gap, -lim, lim), bins=bins, color="0.75",
            label="shuffled control (null)")
    ax.hist(np.clip(g.gap, -lim, lim), bins=bins, histtype="step", lw=2, color=C[a],
            label="real error order")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(g.gap.median(), color=C[a], ls="--", lw=1.6)
    frac = (g.gap < 0).mean()
    ax.set_title(f"{ALGN[a]}\nmedian {g.gap.median():+.2f}%  ·  {frac:.0%} cheaper than shuffled",
                 fontsize=9.5)
    ax.set_xlabel("cost of real order − cost of shuffled orders (%)")
    rows.append({"algorithm": a, "n": len(g), "median_gap_pct": g.gap.median(),
                 "frac_cheaper": frac, "median_sham_gap_pct": g.sham_gap.median(),
                 "sham_frac_cheaper": (g.sham_gap < 0).mean()})
axes[0].set_ylabel("number of error series")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Fig H — identical η₁ and η∞, yet the real temporal order is systematically cheaper"
             "\n900 Azure error series (3 days × 300), 60 permutations each, common random numbers",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(R, "figH_direction.png"), dpi=200, bbox_inches="tight")
pd.DataFrame(rows).to_csv(os.path.join(R, "figH_direction.csv"), index=False)
print(f"[fig] figH_direction.png  | " +
      "  ".join(f"{r['algorithm']}: {r['median_gap_pct']:+.2f}% ({r['frac_cheaper']:.0%} rẻ hơn)"
                for r in rows))

# ────────────────────────────────── Hình I
st = load("verify_c1_stats")
m = b[b.algo == "thr"].merge(st, on=["day", "k"], suffixes=("", "_s"))
fig, ax = plt.subplots(figsize=(6.4, 4.4))
rows = []
for d, mk in zip(DAYS, ["o", "s", "^"]):
    g = m[m.day == d]
    x, y = g.acf_mean_2_30.to_numpy(float), np.clip(g.z.to_numpy(float), -12, 12)
    ax.scatter(x, y, s=12, alpha=.35, marker=mk, color=C["thr"], lw=0, label=f"day {d:02d}")
    q = np.quantile(x, np.linspace(0, 1, 7))
    for i in range(6):
        sel = (x >= q[i]) & (x <= q[i + 1])
        if sel.sum() > 5:
            rows.append({"day": d, "acf_bin_mid": (q[i] + q[i + 1]) / 2,
                         "median_z": np.median(y[sel]), "n": int(sel.sum())})
med = pd.DataFrame(rows).groupby("acf_bin_mid").median_z.mean()
ax.plot(med.index, med.values, color="k", lw=2, marker="o", ms=5, label="median per bin (3 days)")
ax.axhline(0, color="k", lw=1, ls=":")
ax.set_xlabel("persistence of the error series — mean ACF over lags 2–30")
ax.set_ylabel("z of real order vs 60 permutations\n(negative = real order is cheaper)")
ax.set_title("Fig I — the more persistent the error, the cheaper its real order\n"
             "threshold algorithm; partial Spearman ρ = −0.34 to −0.38 after removing\n"
             "permutation-invariant features; replicated on three days", fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(R, "figI_persistence.png"), dpi=200, bbox_inches="tight")
pd.DataFrame(rows).to_csv(os.path.join(R, "figI_persistence.csv"), index=False)
print("[fig] figI_persistence.png")

# ────────────────────────────────── Hình J
oc = load("order_control")
base = (oc[oc.order == "phi+0.00"].groupby(["day", "k"])[[f"cost_{a}" for a in ALG]]
        .mean().rename(columns=lambda c: c + "_base"))
mm = oc.merge(base, on=["day", "k"])
for a in ALG:
    mm[f"rel_{a}"] = 100 * (mm[f"cost_{a}"] - mm[f"cost_{a}_base"]) / mm[f"cost_{a}_base"]
phis = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 0.99]
fig, ax = plt.subplots(figsize=(6.8, 4.4))
rows = []
for a in ALG:
    xs, ys, los, his = [], [], [], []
    for p in phis:
        g = mm[mm.order == f"phi{p:+.2f}"]
        xs.append(g.tau_int.median()); ys.append(g[f"rel_{a}"].median())
        los.append(g[f"rel_{a}"].quantile(.25)); his.append(g[f"rel_{a}"].quantile(.75))
        rows.append({"algorithm": a, "phi": p, "median_tau_int": g.tau_int.median(),
                     "median_rel_cost_pct": g[f"rel_{a}"].median(),
                     "q25": g[f"rel_{a}"].quantile(.25), "q75": g[f"rel_{a}"].quantile(.75)})
    ax.fill_between(xs, los, his, color=C[a], alpha=.12)
    ax.plot(xs, ys, marker="o", ms=5, color=C[a], label=ALGN[a])
    s = mm[mm.order == "sorted"]
    ax.plot([s.tau_int.median()], [s[f"rel_{a}"].median()], marker="*", ms=14, color=C[a])
    rows.append({"algorithm": a, "phi": "sorted", "median_tau_int": s.tau_int.median(),
                 "median_rel_cost_pct": s[f"rel_{a}"].median(),
                 "q25": s[f"rel_{a}"].quantile(.25), "q75": s[f"rel_{a}"].quantile(.75)})
    r = mm[mm.order == "real"]
    ax.plot([r.tau_int.median()], [r[f"rel_{a}"].median()], marker="X", ms=10, color=C[a],
            mec="k", mew=.8, zorder=5)
    rows.append({"algorithm": a, "phi": "real", "median_tau_int": r.tau_int.median(),
                 "median_rel_cost_pct": r[f"rel_{a}"].median(),
                 "q25": r[f"rel_{a}"].quantile(.25), "q75": r[f"rel_{a}"].quantile(.75)})
ax.axhline(0, color="k", lw=1, ls=":")
ax.set_xscale("log")
ax.set_xlabel("imposed persistence of the error order (τ_int, log scale)")
ax.set_ylabel("cost relative to a random order (%)")
ax.set_title("Fig J — imposed persistence does not explain the real order's advantage\n"
             "same error multiset throughout, so η₁ and η∞ are identical; ★ = fully sorted,\n"
             "✗ = the real order, which beats synthetic orders of much higher τ_int", fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(R, "figJ_imposed_order.png"), dpi=200, bbox_inches="tight")
pd.DataFrame(rows).to_csv(os.path.join(R, "figJ_imposed_order.csv"), index=False)
print("[fig] figJ_imposed_order.png")

# ────────────────────────────────── Hình K
op = load("order_position")
base = (op[op.order == "random"].set_index(["day", "k"])[[f"cost_{a}" for a in ALG]]
        .rename(columns=lambda c: c + "_b"))
p = op.merge(base, on=["day", "k"])
for a in ALG:
    p[f"rel_{a}"] = 100 * (p[f"cost_{a}"] - p[f"cost_{a}_b"]) / p[f"cost_{a}_b"]
NAMES = {"sorted_asc": "sorted\nascending",
         "half_small_first": "small half\nfirst",
         "real": "real\norder", "real_reversed": "real order\nreversed",
         "random": "random\n(baseline)",
         "half_large_first": "large half\nfirst",
         "sorted_desc": "sorted\ndescending"}
seq = ["sorted_asc", "half_small_first", "real", "real_reversed", "random",
       "half_large_first", "sorted_desc"]
fig, ax = plt.subplots(figsize=(9.2, 4.6))
w = 0.26
rows = []
for i, a in enumerate(ALG):
    vals = [p[p.order == o][f"rel_{a}"].median() for o in seq]
    ax.bar(np.arange(len(seq)) + (i - 1) * w, vals, width=w, color=C[a], label=ALGN[a])
    for o, v in zip(seq, vals):
        g = p[p.order == o]
        rows.append({"algorithm": a, "order": o, "median_rel_cost_pct": v,
                     "q25": g[f"rel_{a}"].quantile(.25), "q75": g[f"rel_{a}"].quantile(.75),
                     "n": len(g), "median_tau_int": g.tau_int.median()})
ax.axhline(0, color="k", lw=1)
ax.set_xticks(range(len(seq)))
ax.set_xticklabels([NAMES[o] for o in seq], fontsize=8)
ax.set_ylabel("cost relative to a random order (%)")
r = {a: (1 + p[p.order == "sorted_desc"][f"rel_{a}"].median() / 100) /
        (1 + p[p.order == "sorted_asc"][f"rel_{a}"].median() / 100) for a in ALG}
ax.set_title("Fig K — a deterministic counterexample to order-oblivious error measures\n"
             "every bar uses the SAME error multiset, so η₁ and η∞ are identical throughout; "
             f"sorted-descending costs {r['thr']:.1f}× sorted-ascending (threshold)\n"
             "450 Azure error series × 3 days; reversing the real order in time changes nothing "
             "(p ≈ 0.5), so the real-order advantage is not a trend effect", fontsize=9.5)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(R, "figK_counterexample.png"), dpi=200, bbox_inches="tight")
pd.DataFrame(rows).to_csv(os.path.join(R, "figK_counterexample.csv"), index=False)
print("[fig] figK_counterexample.png  | tỉ số desc/asc: " +
      "  ".join(f"{a} {r[a]:.2f}×" for a in ALG))
