"""
Burst — đo cấu trúc thời gian của sai số dự báo trên trace Azure Functions 2019.

Câu hỏi: sai số của forecaster có vón cục theo thời gian không, hay rải đều?

Ba chuỗi được so sánh, tất cả CÙNG phân phối biên của sai số:
  real      : e[t] = |yhat[t] - y[t]| trên chuỗi thật
  ctrl_A    : permutation(e)          -> giữ nguyên eta, phá huỷ mọi cấu trúc.
                                         Vai trò TU TỪ: đây là Figure 1 bằng dữ liệu thật.
  ctrl_B    : permutation(y) -> tính lại e
                                         Vai trò KHOA HỌC: đây là null đúng, vì nó giữ lại
                                         phần tương quan do CHÍNH công thức sai số sinh ra
                                         (e[t] và e[t+1] dùng chung y[t]), chỉ bỏ đi cấu
                                         trúc thời gian của workload.

Kết luận hợp lệ là "acf_real vượt acf_ctrl_B tại lag >= 2", KHÔNG phải "acf_real > 0".
Lag 1 luôn nhiễm artifact chia sẻ số hạng -> báo cáo nhưng không dùng làm bằng chứng.

Chạy:
    env/python.exe src/pipeline.py --smoke        # 20 function, kiểm tra đường ống
    env/python.exe src/pipeline.py                # full 2 tầng x 200 function
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

MINUTE_COLS = [str(i) for i in range(1, 1441)]
MAX_LAG = 60
SEED = 20260913


# ---------------------------------------------------------------- nạp dữ liệu

def _day_path(day):
    path = os.path.join(DATA, f"invocations_per_function_md.anon.d{day:02d}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Chưa có {path}.\nGiải nén: tar -xf data/azurefunctions-dataset2019.tar.xz -C data/"
        )
    return path


def load_day(day=1, nrows=None):
    """Đọc 1 ngày -> (ma trận n_function x 1440, dataframe metadata). Dùng cho --smoke."""
    df = pd.read_csv(_day_path(day), nrows=nrows)
    y = df[MINUTE_COLS].to_numpy(dtype=np.float32)
    meta = df[["HashApp", "HashFunction", "Trigger"]].copy()
    meta["total"] = y.sum(axis=1)
    return y, meta


def scan_totals(day=1, chunksize=5000):
    """
    Lượt 1: chỉ đọc tổng invocation từng function, không giữ ma trận trong RAM.
    File d01 nguyên khối là ~50k x 1440; nạp thẳng float64 tốn ~600 MB chưa kể overhead
    của pandas lúc parse. Hai lượt thì đỉnh RAM chỉ còn cỡ số function được chọn.
    """
    totals, trig = [], []
    for chunk in pd.read_csv(_day_path(day), chunksize=chunksize):
        totals.append(chunk[MINUTE_COLS].to_numpy(dtype=np.float32).sum(axis=1))
        trig.append(chunk["Trigger"].to_numpy())
    meta = pd.DataFrame({"total": np.concatenate(totals), "Trigger": np.concatenate(trig)})
    return meta


def load_rows(day, idx, chunksize=5000):
    """Lượt 2: đọc lại file, chỉ giữ đúng các hàng trong idx, theo thứ tự idx."""
    want = np.sort(np.asarray(idx))
    out, base = [], 0
    for chunk in pd.read_csv(_day_path(day), chunksize=chunksize):
        n = len(chunk)
        sel = want[(want >= base) & (want < base + n)] - base
        if sel.size:
            out.append(chunk.iloc[sel][MINUTE_COLS].to_numpy(dtype=np.float32))
        base += n
    y = np.vstack(out) if out else np.empty((0, 1440), np.float32)
    # trả về theo đúng thứ tự idx ban đầu
    order = np.argsort(np.argsort(np.asarray(idx)))
    return y[order]


def pick_strata(meta, n=200, rng=None):
    """
    Hai tầng mẫu, để câu 'kết quả không phải do cách chọn mẫu' đứng được.
      top    : n function volume cao nhất  (dễ bị nghi selection bias)
      random : n function ngẫu nhiên trong nhóm >= 100 invocation
    """
    rng = rng or np.random.default_rng(SEED)
    total = meta["total"].to_numpy()
    order = np.argsort(-total)
    top = order[:n]

    eligible = np.flatnonzero(total >= 100)
    eligible = np.setdiff1d(eligible, top)
    size = min(n, len(eligible))
    rand = rng.choice(eligible, size=size, replace=False) if size else np.array([], int)

    return {"top": top, "random": rand}


# ---------------------------------------------------------------- forecaster

def forecast(y, kind):
    """
    Trả về (yhat, warmup). yhat[:, t] chỉ có nghĩa với t >= warmup.
    Cố tình dùng forecaster ngây thơ: chúng cho CẬN DƯỚI của burstiness, và
    không có tham số nào để bị cáo buộc là đã tinh chỉnh cho ra kết quả mong muốn.
    """
    if kind == "naive":                      # y_hat[t] = y[t-1]
        return np.roll(y, 1, axis=1), 1
    if kind == "seasonal60":                 # y_hat[t] = y[t-60]
        return np.roll(y, 60, axis=1), 60
    if kind == "ma10":                       # trung bình trượt 10 bước
        c = np.cumsum(np.pad(y, ((0, 0), (1, 0))), axis=1)
        ma = (c[:, 10:] - c[:, :-10]) / 10.0
        out = np.zeros_like(y)
        out[:, 10:] = ma[:, :-1]
        return out, 10
    raise ValueError(kind)


def error_series(y, kind):
    yhat, w = forecast(y, kind)
    return np.abs(yhat - y)[:, w:]


# ---------------------------------------------------------------- ACF

def acf_matrix(x, max_lag=MAX_LAG):
    """
    ACF cho từng hàng, ước lượng biased (chia cho cùng mẫu số) như quy ước.
    Trả về (n_series, max_lag+1). Hàng hằng số -> NaN, lọc ở tầng trên.
    """
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(axis=1, keepdims=True)
    denom = (xc ** 2).sum(axis=1)
    out = np.full((x.shape[0], max_lag + 1), np.nan)
    ok = denom > 0
    out[ok, 0] = 1.0
    for k in range(1, max_lag + 1):
        num = (xc[:, :-k] * xc[:, k:]).sum(axis=1)
        out[ok, k] = num[ok] / denom[ok]
    return out


def tau_int(rho):
    """
    Integrated autocorrelation time: tau = 1 + 2*sum_{k=1..K} rho_k,
    cắt ở K đầu tiên có rho_K < 0 (Geyer initial-positive-sequence).

    Dùng cái này làm SỐ CHÍNH thay cho tau_mix của chain 2 trạng thái, vì chain 2
    trạng thái ngụ ý độ dài burst phân phối hình học — mà đó đúng là thứ Hình B
    định bác bỏ. Dùng nó làm số chính là tự mâu thuẫn.
    """
    out = np.empty(rho.shape[0])
    for i, r in enumerate(rho):
        s, k = 0.0, 1
        while k < r.shape[0] and np.isfinite(r[k]) and r[k] > 0:
            s += r[k]
            k += 1
        out[i] = 1.0 + 2.0 * s
    return out


# ---------------------------------------------------------------- burst

def burst_lengths(e, q=90):
    """Ngưỡng phân vị q TÍNH RIÊNG từng function -> độ dài các đoạn liên tiếp vượt ngưỡng."""
    lengths = []
    for row in e:
        thr = np.percentile(row, q)
        b = row > thr
        if not b.any():
            continue
        d = np.diff(np.concatenate(([0], b.view(np.int8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        lengths.append(ends - starts)
    return np.concatenate(lengths) if lengths else np.array([], int)


# ---------------------------------------------------------------- thí nghiệm

def run(y_sub, forecaster, rng):
    """Trả về dict 3 biến thể -> (acf trung bình qua function, tau_int từng function)."""
    e_real = error_series(y_sub, forecaster)

    # Control A: xáo trực tiếp chuỗi sai số (eta bảo toàn chính xác)
    e_a = np.apply_along_axis(rng.permutation, 1, e_real)

    # Control B: xáo tín hiệu gốc rồi TÍNH LẠI sai số qua đúng forecaster đó
    y_shuf = np.apply_along_axis(rng.permutation, 1, y_sub)
    e_b = error_series(y_shuf, forecaster)

    out = {}
    for name, e in (("real", e_real), ("ctrl_A", e_a), ("ctrl_B", e_b)):
        rho = acf_matrix(e)
        good = np.isfinite(rho[:, 1])
        out[name] = {
            "acf_mean": np.nanmean(rho[good], axis=0),
            "acf_q25": np.nanpercentile(rho[good], 25, axis=0),
            "acf_q75": np.nanpercentile(rho[good], 75, axis=0),
            "tau": tau_int(rho[good]),
            "n": int(good.sum()),
            "eta_sum": float(e.sum()),
            "burst": burst_lengths(e),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="20 function, chỉ để kiểm đường ống")
    ap.add_argument("--forecasters", default="naive,seasonal60")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(SEED)

    if args.smoke:
        y_all, meta = load_day(args.day, nrows=5000)
        n = 20
        print(f"[load] d{args.day:02d} SMOKE  shape={y_all.shape}  "
              f"tổng invocation={meta['total'].sum():,.0f}")
    else:
        y_all, n = None, args.n
        meta = scan_totals(args.day)
        print(f"[scan] d{args.day:02d}  n_function={len(meta):,}  "
              f"tổng invocation={meta['total'].sum():,.0f}")

    strata = pick_strata(meta, n=n, rng=rng)
    rows = []
    for stratum, idx in strata.items():
        if len(idx) == 0:
            continue
        y_sub = y_all[idx] if y_all is not None else load_rows(args.day, idx)
        print(f"[load] tầng {stratum}: {y_sub.shape[0]} function")
        for fc in args.forecasters.split(","):
            res = run(y_sub, fc, rng)
            for variant, r in res.items():
                rows.append({
                    "stratum": stratum, "forecaster": fc, "variant": variant,
                    "n_func": r["n"], "eta_sum": r["eta_sum"],
                    "tau_int_median": float(np.median(r["tau"])),
                    "tau_int_q25": float(np.percentile(r["tau"], 25)),
                    "tau_int_q75": float(np.percentile(r["tau"], 75)),
                    "acf_lag1": float(r["acf_mean"][1]),
                    "acf_lag2": float(r["acf_mean"][2]),
                    "acf_lag5": float(r["acf_mean"][5]),
                    "acf_lag10": float(r["acf_mean"][10]),
                    "burst_mean": float(r["burst"].mean()) if r["burst"].size else np.nan,
                    "burst_p99": float(np.percentile(r["burst"], 99)) if r["burst"].size else np.nan,
                    "burst_max": int(r["burst"].max()) if r["burst"].size else 0,
                })
            tag = f"{stratum}_{fc}"
            np.savez(os.path.join(RESULTS, f"acf_{tag}.npz"),
                     **{f"{v}_{k}": res[v][k]
                        for v in res for k in ("acf_mean", "acf_q25", "acf_q75", "tau", "burst")})
            print(f"[run ] {tag:24s} n={res['real']['n']:4d}  "
                  f"acf2 real={res['real']['acf_mean'][2]:+.3f} "
                  f"A={res['ctrl_A']['acf_mean'][2]:+.3f} "
                  f"B={res['ctrl_B']['acf_mean'][2]:+.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "summary_smoke.csv" if args.smoke else "summary.csv")
    df.to_csv(out, index=False)
    print(f"\n[save] {out}")

    # ---- Gate 2: đọc thẳng ra kết luận, không cần nhìn hình
    print("\n=== GATE 2 — acf(real) có vượt acf(ctrl_B) tại lag >= 2 không? ===")
    for (s, f), g in df.groupby(["stratum", "forecaster"]):
        gi = g.set_index("variant")
        gaps = {L: gi.loc["real", f"acf_lag{L}"] - gi.loc["ctrl_B", f"acf_lag{L}"]
                for L in (2, 5, 10)}
        verdict = "H1 SỐNG" if all(v > 0.02 for v in gaps.values()) else "cần xem lại"
        print(f"  {s:7s} {f:12s} " +
              "  ".join(f"lag{L}: {v:+.3f}" for L, v in gaps.items()) + f"   -> {verdict}")


if __name__ == "__main__":
    main()
