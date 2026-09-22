# Nhật ký quá trình huấn luyện — RQ1: So sánh ba mô hình cơ sở (TOP / AdaLEA / RiskProp)

**Ngày thực hiện:** 22/09/2026
**Môi trường tính toán:** vast.ai — 1× GPU NVIDIA RTX 4070 Ti SUPER (16GB VRAM, 42 TFLOPS), truy cập qua Jupyter Terminal
**Dữ liệu:** Bộ dữ liệu chính thức Nexar Collision Prediction trên HuggingFace (`nexar-ai/nexar_collision_prediction`), 1.500 video huấn luyện (750 video có va chạm / 750 video bình thường)

Tài liệu này ghi lại toàn bộ quy trình thiết lập môi trường, huấn luyện và đánh giá ba mô hình cơ sở phục vụ câu hỏi nghiên cứu RQ1, cùng với các mốc thời gian thực tế và những vấn đề kỹ thuật phát sinh trong quá trình thực hiện, nhằm phục vụ việc theo dõi tiến độ và tái lập kết quả cho các thành viên trong nhóm.

## 1. Thiết lập môi trường

| Bước | Nội dung thực hiện | Thời gian |
|---|---|---|
| Sao chép mã nguồn | Clone repository từ GitHub (`TranMinhHuy15/CPV301`) | < 1 phút |
| Cài đặt thư viện | Cài các thư viện cần thiết: decord, pytorchvideo, scikit-learn, pandas, pillow, huggingface_hub | ~2 phút |
| Tải dữ liệu | Tải bộ dữ liệu Nexar Collision Prediction từ HuggingFace (dung lượng 31,4GB) | ~24 phút |
| Chuẩn hóa nhãn dữ liệu | Chuyển đổi file metadata gốc của HuggingFace (định dạng videofolder, tách theo hai thư mục positive/negative) thành file nhãn thống nhất (id, nhãn, thời điểm va chạm, thời điểm cảnh báo) | ~1 phút |
| Điều chỉnh đường dẫn mã nguồn | Cập nhật lại toàn bộ đường dẫn trong mã nguồn cho phù hợp với môi trường máy chủ mới | < 1 phút |
| Bổ sung mã nguồn còn thiếu | Khôi phục hai tệp mã nguồn của mô hình TOP (`cell11b_cached_dataset.py`, `cell13_model.py`) vốn đã được kiểm chứng hoạt động ở giai đoạn thử nghiệm trước đó nhưng chưa được lưu trữ độc lập trên repository | ~5 phút |

**Tổng thời gian thiết lập:** khoảng 35 phút.

## 2. Mô hình TOP (5-frame)

| Giai đoạn | Mô tả | Thời gian |
|---|---|---|
| Tiền xử lý dữ liệu | Trích xuất và lưu cache các đoạn video 5-khung hình (causal window) | 8,0 phút |
| Huấn luyện | 50 epoch, SGD (lr=0,01), giảm learning rate theo lịch trình tại epoch 20 và 40 | 7,7 phút — mô hình tốt nhất đạt tại epoch 4 |
| Đánh giá | Đánh giá đa mốc thời gian cảnh báo (0,5s / 1,0s / 1,5s) | ~1 phút |

**Kết quả:** Proposal mAP = 0,6496 · Video-level AUC = 0,6516 · mTTA (FAR≤0,1) = 1,114s

## 3. Mô hình AdaLEA (5-frame)

Mô hình AdaLEA tái sử dụng bộ cache dữ liệu đã xây dựng cho mô hình TOP, không cần tiền xử lý riêng.

| Giai đoạn | Mô tả | Thời gian |
|---|---|---|
| Huấn luyện | 50 epoch, cơ chế ngưỡng cảnh báo thích ứng (adaptive threshold) | 11,9 phút — mô hình tốt nhất đạt tại epoch 2 |
| Đánh giá | Đánh giá đa mốc thời gian cảnh báo | ~1 phút |

**Kết quả:** Proposal mAP = 0,6931 · Video-level AUC = 0,6860 · mTTA (FAR≤0,1) = 1,092s

## 4. Mô hình RiskProp

Do đặc thù kiến trúc (yêu cầu chuỗi khung hình có thứ tự thời gian để tính toán các hàm mất mát FFR và AMC), mô hình RiskProp cần một bộ cache dữ liệu riêng.

| Giai đoạn | Mô tả | Thời gian |
|---|---|---|
| Tiền xử lý dữ liệu | Trích xuất chuỗi 12 đoạn video/video gốc, bước nhảy 0,5 giây | 12,2 phút |
| Huấn luyện | 50 epoch, kết hợp hàm mất mát FFR (Future-Frame Regularization) và AMC (Adaptive Monotonic Constraint) | 81,9 phút — mô hình tốt nhất đạt tại epoch 15 |
| Đánh giá | Đánh giá đa mốc thời gian cảnh báo | ~1 phút |

**Kết quả:** Proposal mAP = 0,6749 · Video-level AUC = 0,7124 · mTTA (FAR≤0,1) = 1,266s

## 5. Tổng kết thời gian và chi phí

- **Tổng thời gian thực hiện** (thiết lập môi trường và huấn luyện cả ba mô hình): xấp xỉ 2,5 giờ
- **Chi phí thuê GPU:** khoảng 0,51 USD (trong phạm vi ngân sách 5 USD đã dự trù)

## 6. Các vấn đề kỹ thuật phát sinh và hướng xử lý

1. **Cấu trúc thư mục trên GitHub bị làm phẳng** (không phân theo thư mục con của từng mô hình) do giới hạn thao tác kéo-thả khi tải lên qua giao diện web của GitHub. Do tên các tệp không trùng lặp, cấu trúc phẳng vẫn đảm bảo mã nguồn hoạt động chính xác, nhóm quyết định giữ nguyên để tiết kiệm thời gian.
2. **Thiếu hai tệp mã nguồn của mô hình TOP** do các tệp này được phát triển và kiểm thử trực tiếp trên Kaggle ở phiên làm việc trước nhưng chưa được xuất ra tệp độc lập. Hai tệp đã được khôi phục lại dựa trên việc đối chiếu logic với tệp đánh giá đã có sẵn, đảm bảo tính nhất quán với toàn bộ pipeline.
3. **Dao động bất thường trong quá trình huấn luyện RiskProp** tại epoch 21–22 (giá trị mất mát trên tập kiểm định tăng đột biến), nhiều khả năng do dao động gradient tạm thời. Vấn đề này không ảnh hưởng đến kết quả cuối cùng vì cơ chế lưu checkpoint tốt nhất đã chọn được mô hình tại epoch 15 (thời điểm trước khi xảy ra dao động).
4. **Yêu cầu xác thực bằng Personal Access Token khi đẩy mã nguồn lên GitHub**, do GitHub đã ngừng hỗ trợ xác thực bằng mật khẩu thông thường qua giao thức HTTPS.

## 7. Phạm vi chưa thực hiện trong phiên làm việc này

- Đánh giá trên tập kiểm tra chính thức của Nexar (với nhãn thật từ HuggingFace, `solution.csv`) chưa được thực hiện. Hiện tại, cả ba mô hình đang được đánh giá trên tập validation tách từ dữ liệu huấn luyện (300/1.500 video), nhằm đảm bảo tính công bằng khi so sánh nội bộ giữa ba mô hình trong khuôn khổ RQ1. Đây là một hướng mở rộng có thể thực hiện ở giai đoạn sau để đối chiếu với kết quả công bố trong các bài báo gốc.
- Nghiên cứu RQ2 (đánh giá phân rã bốn thành phần của RiskProp) và RQ3 (so sánh chiến lược ghép cặp cố định và ngẫu nhiên trong hàm mất mát AMC) là các nội dung nằm ngoài phạm vi của phiên làm việc này, dự kiến triển khai ở các bước tiếp theo của đề tài.
- Các checkpoint mô hình tốt nhất (`best_*.pth`) hiện đang được lưu trữ trên máy chủ vast.ai và sẽ được tải về lưu trữ cục bộ trước khi kết thúc phiên thuê máy chủ.

## Tài liệu liên quan (thư mục `results/`)

- `RQ1_comparison.md` — Bảng so sánh chi tiết và phân tích kết quả ba mô hình
- `top5f_training_log.csv`, `top5f_eval_results.txt` — Nhật ký huấn luyện và kết quả đánh giá mô hình TOP
- `adalea5f_training_log.csv`, `adalea5f_eval_results.txt` — Nhật ký huấn luyện và kết quả đánh giá mô hình AdaLEA
- `riskprop_training_log.csv`, `riskprop_eval_results.txt` — Nhật ký huấn luyện và kết quả đánh giá mô hình RiskProp
