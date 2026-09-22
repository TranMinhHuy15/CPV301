# RiskProp v2 — Kết quả tối ưu (multi-seed 42/43/44)

Sau khi áp dụng các thay đổi theo góp ý của partner (xem `RISKPROP_V2_GUIDE.md`), RiskProp v2 được train độc lập 3 lần với seed 42, 43, 44 để kiểm chứng độ ổn định thống kê.

## Kết quả từng seed

| Chỉ số | Seed 42 | Seed 43 | Seed 44 | Trung bình ± Độ lệch chuẩn |
|---|---|---|---|---|
| Proposal mAP | 0.7646 | 0.7658 | 0.7464 | **0.7589 ± 0.0089** |
| mAUC@0.1 (mean 3 horizon) | 0.2388 | 0.2579 | 0.2501 | **0.2489 ± 0.0078** |
| mTTA@FAR<=0.1 (detected) | 1.115s | 1.109s | 1.092s | **1.105s ± 0.010s** |
| Coverage | 64.0% | 67.3% | 65.3% | **65.5% ± 1.4%** |
| Video-level AUC | 0.7797 | 0.7774 | 0.7674 | **0.7748 ± 0.0053** |
| Epoch dừng (early-stop) | 35 | 25 | 25 | — |
| Best checkpoint epoch | 29 | 19 | 19 | — |
| Thời gian train | 56.9 phút | 41.0 phút | 41.0 phút | — |

## Đối chiếu với tiêu chí thành công (do partner đề ra)

| Chỉ số | Ngưỡng yêu cầu | Trung bình 3 seed | Kết luận |
|---|---|---|---|
| mAP | > 0.6931 (AdaLEA) | 0.7589 | ✅ Đạt (vượt cả 3/3 seed) |
| mAUC@0.1 | > 0.1920 (AdaLEA) | 0.2489 | ✅ Đạt (vượt cả 3/3 seed) |
| mTTA (detected) | >= 1.10s | 1.105s | ✅ Đạt về trung bình (seed 44 hụt nhẹ, 1.092s, dao động ngẫu nhiên bình thường giữa các seed) |
| Coverage | >= 47.3% (AdaLEA) | 65.5% | ✅ Đạt (vượt cả 3/3 seed) |

**Kết luận chung**: RiskProp v2 vượt AdaLEA-5f trên mAP, mAUC@0.1, coverage một cách ổn định qua cả 3 lần train độc lập (khác seed), đạt yêu cầu thành công partner đặt ra trong "KE HOACH KHAC PHUC VA TOI UU RISKPROP".

## So sánh v1 (trước tối ưu) vs v2 (sau tối ưu, trung bình 3 seed)

| Chỉ số | RiskProp v1 | RiskProp v2 (TB 3 seed) | Thay đổi |
|---|---|---|---|
| Proposal mAP | 0.6749 | 0.7589 | +0.084 (+12.4%) |
| mAUC@0.1 | 0.1388 | 0.2489 | +0.110 (+79.3%) |
| mTTA (detected) | 1.266s | 1.105s | -0.161s (giảm, nhưng vẫn đạt ngưỡng) |
| Coverage | 41.3% | 65.5% | +24.2 điểm % |
| Video AUC | 0.7124 | 0.7748 | +0.062 |

## Các thay đổi kỹ thuật (v1 -> v2)

- Learning rate: 0.01 -> 0.002
- lambda_reg (FFR) / lambda_mono (AMC): 1.5/1.1 -> 0.5/0.5
- collision_weight: 5.0 -> 8.0
- AMC pairs/video: 4 -> 8
- Temporal pairing: random offset -> fixed lag 1.0s
- FFR: tính trên raw logit -> tính trên sigmoid probability
- Thêm gradient clipping (max norm 1.0)
- Checkpoint + early stopping: theo val_loss -> theo val mAP (patience 7 epoch)

## File liên quan

- `eval_results_rq1_riskprop_seed{42,43,44}.txt` — kết quả eval chi tiết từng seed
- `training_log_riskprop_seed{42,43,44}.csv` — log training đầy đủ từng epoch
- `riskprop_v1_eval_results.txt`, `riskprop_v1_training_log.csv` — kết quả v1 (trước tối ưu), giữ lại để đối chiếu
- Checkpoint (`best_riskprop_seed{42,43,44}.pth`): không lưu git (quá nặng), tải riêng từ Jupyter file browser trên vast.ai instance.
