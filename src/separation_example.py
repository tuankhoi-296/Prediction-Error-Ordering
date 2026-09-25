"""
Phản ví dụ TƯỜNG MINH — phép dựng, không phải đo đạc.

Mục 7.4 của KETQUA-D1 cho thấy sắp xếp tăng dần vs giảm dần lệch 2,38× trên dữ liệu Azure.
Đó là kết quả thực nghiệm. File này dựng một instance tối giản có thể phân tích bằng tay, và
xác nhận con số bằng chính simulator đã dùng cho mọi thí nghiệm khác.

  Instance:  n khách hàng, TẤT CẢ tại điểm 0. Chi phí mở facility f.
  Đa tập sai số:  {0 lặp n/2 lần} ∪ {M lặp n/2 lần}.   Dự đoán p_i = 0 + d_i.

  Thứ tự A (tăng dần, các số 0 trước):
      bước 1      : dự đoán 0  → mở facility tại 0 (đúng chỗ)
      bước n/2+1  : dự đoán M  → mở thêm facility tại M (thừa, tốn f)
      chi phí phục vụ = 0 vì facility tại 0 luôn tồn tại
      TỔNG = 2f

  Thứ tự B (giảm dần, các số M trước):
      bước 1      : dự đoán M  → mở facility tại M. KHÔNG RÚT LẠI ĐƯỢC.
      n/2 bước đầu: khách tại 0 phải dùng facility tại M → mỗi khách tốn M
      bước n/2+1  : dự đoán 0  → mở facility tại 0
      TỔNG = 2f + (n/2)·M

  η₁ = (n/2)·M và η∞ = M ở CẢ HAI thứ tự. OPT = f ở cả hai (một facility tại 0).
  Tỉ số chi phí = 1 + n·M/(4f)  →  không bị chặn.

Phạm vi: lập luận đúng cho thuật toán MỞ FACILITY TẠI VỊ TRÍ ĐƯỢC DỰ ĐOÁN (threshold và họ
Meyerson đều vậy). Thuật toán bỏ qua dự đoán hoàn toàn thì không dính — đúng như số đo ở ε = 0.
Vì vậy phát biểu tổng quát phải nói cho lớp thuật toán CÓ CONSISTENCY, và đó là Milestone 1.

    hd/python.exe src/separation_example.py
"""

import numpy as np

from ofl_cost2 import simulate
from robust_c1 import simulate_meyerson


def build(n, M):
    """Trả về (demands, thứ tự tăng dần, thứ tự giảm dần). Hai thứ tự cùng một đa tập."""
    x = np.zeros(n)
    asc = np.concatenate([np.zeros(n // 2), np.full(n // 2, float(M))])
    return x, asc, asc[::-1].copy()


def main():
    f, theta = 10.0, 8.0
    print(f"{'n':>6}{'M':>6}{'η₁':>9}{'η∞':>5}{'OPT':>6}"
          f"{'chi phí tăng dần':>18}{'chi phí giảm dần':>18}{'tỉ số':>9}{'lý thuyết':>11}")
    for n, M in [(100, 20), (200, 50), (400, 50), (200, 200), (1000, 100)]:
        x, asc, desc = build(n, M)
        ca, ka = simulate(x, x + asc, f, theta)
        cd, kd = simulate(x, x + desc, f, theta)
        theory_ratio = 1.0 + n * M / (4.0 * f)
        assert abs(ca - 2 * f) < 1e-9, ca
        assert abs(cd - (2 * f + (n // 2) * M)) < 1e-9, cd
        assert asc.sum() == desc.sum() and asc.max() == desc.max()
        print(f"{n:>6}{M:>6}{asc.sum():>9.0f}{asc.max():>5.0f}{f:>6.0f}"
              f"{ca:>18.1f}{cd:>18.1f}{cd / ca:>9.1f}×{theory_ratio:>10.1f}×")

    print("\nHọ Meyerson (ngẫu nhiên, trung bình 200 lần chạy) — cùng hiện tượng:")
    x, asc, desc = build(200, 50)
    for name, d in [("tăng dần", asc), ("giảm dần", desc)]:
        c = np.mean([simulate_meyerson(x, x + d, f, np.random.default_rng(s))[0]
                     for s in range(200)])
        print(f"  {name:10s} chi phí {c:9.1f}   competitive ratio {c / f:7.1f}")

    print("\nĐối chứng: thuật toán BỎ QUA dự đoán (mở tại vị trí thật) không dính phản ví dụ:")
    for name, d in [("tăng dần", asc), ("giảm dần", desc)]:
        c, k = simulate(x, x + 0 * d, f, theta)
        print(f"  {name:10s} chi phí {c:9.1f}  → độ nhạy thứ tự = 1.0, đúng như đo ở ε = 0")


if __name__ == "__main__":
    main()
