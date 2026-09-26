# RQ1b: RiskProp-full (code gốc của tác giả) trên Nexar

Thí nghiệm bổ sung cho RQ1: **kiểm tra xem bản RiskProp của nhóm có bị thiệt do cách cài đặt hay không**, bằng cách train RiskProp bằng chính code công khai của tác giả, rồi đánh giá theo đúng protocol RQ1. Mục tiêu không phải là làm cho RiskProp thắng. Kết quả ra sao cũng báo cáo.

- RQ1 hiện tại vẫn là kết quả chính. TOP và AdaLEA không train lại.
- RQ2 và RQ3 giữ nguyên và được ghi rõ là biến thể của **RiskProp bản cũ**.
- Official test chỉ chạy **một lần** cho RQ1b, sau khi đã chốt checkpoint trên val. Trong bài phải khai báo đây là lần dùng test thứ 2.

## 1. Nguồn và các điểm lệch (deviation)

Code của tác giả lấy từ `github.com/xingyueye5/RiskProp`, commit `579376f`, giữ nguyên config `configs/predict_anomaly_snippet.py` và `configs/_base_/schedules/sgd_50e.py`.

| # | Điểm lệch | Lý do |
|---|---|---|
| 1 | Sửa 3 lỗi để code import được (`patch_authors_code.py`, diff trong `authors_code_fixes.diff`): thiếu dấu phẩy trong `taa/models.py`; `taa/__init__.py` không import `models`; `taa/__init__.py` import module `model_AdaLEA` không có trong repo | Code công khai **không chạy được như đang phát hành**. Chỉ sửa đúng 3 chỗ này, không đổi logic. |
| 2 | Tách frame ở **10 fps** thay vì 30 fps. `accident_frame = round(time_of_event × 10)`, `abnormal_start_frame = round(time_of_alert × 10)`, loader dùng `fps = 10` | Tiết kiệm đĩa. Tác giả cũng lấy mẫu ở 10 fps (frame_interval 3 trên video 30 fps), nên clip vẫn gồm 5 frame cách nhau 0,1 s và 30 clip vẫn phủ 3,3 s. Mất phần xê dịch ngẫu nhiên 0–2 frame gốc (dưới 0,1 s) của clip dương khi train. |
| 3 | Split val = 300 video của nhóm (`results/repro/split_manifest_seed42.json`) thay cho `nexar_val` của tác giả | Để so sánh được với TOP và AdaLEA. Split của tác giả chỉ trùng 48/300 video với split của nhóm. |
| 4 | 1 GPU với gradient accumulation 8 (2 video × 8 = 16 video mỗi bước cập nhật) | Tác giả dùng nhiều GPU. README ghi 8 GPU, còn `dist_train.sh` ghi 6 GPU. BatchNorm vẫn thấy 2 video × 30 clip mỗi lượt forward, giống trên mỗi GPU của tác giả. |
| 5 | Frame lưu với cạnh ngắn 256 px | Tiết kiệm đĩa. Pipeline của tác giả vẫn áp RandomResizedCrop rồi resize về 224×224 như config. |
| 6 | Chọn checkpoint theo luật đã khóa của nhóm: chọn giữa epoch có val loss thấp nhất và epoch cuối, lấy cái có mAP 3 mốc cao hơn. Val loss ở đây là BCE (pos_weight 1) trên window 1,0 s | Giống RQ1. Checkpoint theo luật của tác giả (best mAUC@) được báo cáo ở phụ lục, **chỉ trên val**. |
| 7 | *(chỉ khi phải dùng)* `AMP=1` (mixed precision) | Chỉ dùng khi GPU không đủ bộ nhớ cho fp32. Phải ghi là deviation. |
| 8 | *(chỉ khi phải dùng)* mmcv 2.1.0 thay vì 2.2.0 | Chỉ khi không có wheel 2.2.0. `setup_env.sh` sẽ báo. |

## 2. Chuẩn bị trước khi thuê máy

1. Upload thư mục `riskprop_full/` lên GitHub (repo CPV301).
2. Nạp credit vast. Xem bảng chi phí ở mục 5.
3. Tạo **token Hugging Face mới** (Read) và **token GitHub mới**. Chỉ gõ trong terminal của máy thuê, **không dán vào chat**.
4. Để sẵn trên máy mình 6 checkpoint TOP/AdaLEA đã được chọn: `latest_cached_seed42/43/44.pth`, `best_adalea_seed42.pth`, `latest_adalea_seed43.pth`, `latest_adalea_seed44.pth`.

**Chọn GPU:**
- Nên dùng **A100 80 GB hoặc H100 80 GB**. Mỗi bước forward có 60 clip (2 video × 30), và theo ước tính chạy fp32 cần khoảng 80 GB, giống máy A800 80 GB của tác giả.
- Nếu chỉ có GPU 48 GB (A6000, L40S, A40) thì bắt buộc dùng `AMP=1` (deviation 7).
- Không dùng GPU 24 GB.
- Ổ đĩa: **200 GB**.

## 3. Các bước trên máy vast

Mỗi bước có phần kiểm tra. Bước nào kiểm tra không đạt thì **dừng lại và báo**.

```bash
# 3.1 Lấy code + dữ liệu Nexar (môi trường mặc định của máy, như lần trước)
cd /workspace && git clone https://github.com/TranMinhHuy15/CPV301.git && cd CPV301
source /venv/main/bin/activate            # nếu image có venv này
export HF_TOKEN=<token Hugging Face>      # gõ trực tiếp trong terminal
python pipeline/cell09_prepare_hf_data.py # tạo data/nexar_kaggle_style/

# 3.2 Môi trường riêng cho code tác giả (khoảng 10-15 phút)
bash riskprop_full/setup_env.sh
# KIỂM TRA: dòng cuối in "OK" và "Environment ready"; mmcv 2.2.0 (nếu hiện 2.1.0 thì ghi deviation 8)

# Từ đây trở đi, mỗi terminal mới đều chạy:
source /workspace/rq1b_env/bin/activate
export PYTHONPATH=/workspace/CPV301/riskprop_full:$PYTHONPATH

# 3.3 Tách frame 10 fps + tạo annotations.csv (khoảng 15-30 phút)
python /workspace/CPV301/riskprop_full/prepare_nexar_10fps.py --workers 8
# KIỂM TRA: 1500 video (750 dương / 750 âm), failed = 0, số video bị clamp gần 0;
#   3 dòng "check": accident_frame/10 khớp time_of_event

# 3.4 Cache val của nhóm (re01, giống RQ1)
cd /workspace/CPV301 && python reeval/re01_precache_val.py

# 3.5 Kiểm tra dữ liệu (bắt buộc PASS hết)
cd /workspace/RiskProp
python /workspace/CPV301/riskprop_full/check_rq1b_setup.py configs/riskprop_full_nexar.py
# KIỂM TRA: "N/N checks passed". Mở rq1b_check_clips.png: clip cuối phải là lúc va chạm
```

**3.6 Chạy thử 1 epoch** (seed 42, thư mục riêng, không tính là kết quả):
```bash
cd /workspace/RiskProp
python tools/train.py configs/riskprop_full_nexar.py --work-dir work_dirs/rq1b_trial \
    --cfg-options train_cfg.max_epochs=1 randomness.seed=42 2>&1 | tee trial.log
# Terminal khác: watch -n 5 nvidia-smi   (theo dõi bộ nhớ GPU)
```
Kiểm tra:
- Trong `trial.log` có dòng `Load checkpoint from https://download.openmmlab.com/...kinetics710...`, tức weights Kinetics-710 đã được nạp.
- Loss (`loss_cls`, `loss_ffr`, `loss_mono`) là số hữu hạn, không phải `nan`.
- Không bị lỗi thiếu bộ nhớ (out of memory). Nếu bị, chạy lại với `--amp` và ghi deviation 7.
- Ghi lại thời gian của 1 epoch để ước tính tổng thời gian.

Sau đó xóa thư mục thử: `rm -rf work_dirs/rq1b_trial`. Nếu phải **sửa code hoặc config** sau bước này thì báo nhóm trước khi chạy tiếp.

**3.7 Train chính thức 3 seed** (trong tmux, khoảng 8–10 giờ trên A100):
```bash
tmux new -s rq1b
bash /workspace/CPV301/riskprop_full/run_rq1b_train.sh          # hoặc: AMP=1 bash ...
# Tắt terminal an toàn; mở lại: tmux attach -t rq1b
```

**3.8 Dự đoán val của TOP/AdaLEA**, cần cho CI ghép cặp. Upload 6 checkpoint ở mục 2 vào `/workspace/CPV301/ckpts/`, rồi chạy:
```bash
cd /workspace/CPV301
python reeval/re02_infer_val.py --ckpt-dir /workspace/CPV301/ckpts \
    --only top_seed42,top_seed43,top_seed44,adalea_seed42,adalea_seed43,adalea_seed44
```

**3.9 Chốt checkpoint trên val, rồi push lên GitHub TRƯỚC khi chạy test.** Commit này là bằng chứng có ngày giờ cho thấy checkpoint đã được chốt trước.
```bash
cd /workspace/RiskProp
python /workspace/CPV301/riskprop_full/infer_rq1b.py --stage val
python /workspace/CPV301/riskprop_full/analyze_rq1b.py --stage val
# KIỂM TRA: trong summary_rq1b_val.md có 2 dòng PASS (khớp per_run_metrics.csv và bootstrap_ci.csv của RQ1)
mkdir -p /workspace/CPV301/results/rq1b && cp -r /workspace/CPV301/reeval_out/rq1b/{rq1b_chosen_checkpoints.json,rq1b_val_per_epoch_seed*.csv,analysis_val,preds_val_rq1b} /workspace/CPV301/results/rq1b/
cd /workspace/CPV301 && git add results/rq1b riskprop_full/environment_rq1b.txt riskprop_full/authors_code_fixes.diff && git commit -m "RQ1b: validation results, checkpoints locked before test" && git push
```

**3.10 Official test: chạy ĐÚNG MỘT LẦN.** Nếu đang ở terminal mới, nhớ `export HF_TOKEN=...` lại. Script sẽ tự từ chối nếu phát hiện submission RQ1b đã tồn tại.
```bash
cd /workspace/RiskProp
python /workspace/CPV301/riskprop_full/infer_rq1b.py --stage test --download
python /workspace/CPV301/riskprop_full/analyze_rq1b.py --stage test
# KIỂM TRA: 2 dòng PASS ở cuối (21 run cũ và các CI cũ được tái lập y hệt)
```

**3.11 Lưu kết quả lên GitHub, tải checkpoint về, rồi destroy máy**
```bash
cp -r /workspace/CPV301/reeval_out/rq1b/{analysis_test,test_infer_rq1b} /workspace/CPV301/results/rq1b/
mkdir -p /workspace/CPV301/results/rq1b/logs && cp /workspace/RiskProp/logs_rq1b/*.log /workspace/RiskProp/data/nexar-collision-prediction/prepare_meta.json /workspace/RiskProp/rq1b_check_clips.png /workspace/CPV301/results/rq1b/logs/
cd /workspace/CPV301 && git add results/rq1b && git commit -m "RQ1b: official test (single run) + logs" && git push
```
- Tải về máy: checkpoint đã chọn cho mỗi seed (xem `rq1b_chosen_checkpoints.json`) và `best_mAUC@_epoch_*.pth`, tổng cộng 6 file hoặc ít hơn.
- **Không** tải cả 150 file epoch.
- Kiểm tra GitHub đủ file rồi mới destroy máy (biểu tượng thùng rác, không bấm Stop). Xóa token HF và GitHub sau khi xong.

## 4. Các file trong thư mục này

| File | Việc |
|---|---|
| `patch_authors_code.py` | Sửa 3 lỗi import trong code tác giả, ghi diff ra `authors_code_fixes.diff` |
| `nexar_rq1b.py` | Dataset Nexar = loader của tác giả với fps 10 và split của nhóm |
| `configs/riskprop_full_nexar.py` | Kế thừa nguyên config của tác giả; chỉ đổi dataset, gradient accumulation, lưu mọi epoch |
| `prepare_nexar_10fps.py` | Tách frame 10 fps và tạo `annotations.csv` |
| `check_rq1b_setup.py` | Kiểm tra split, 30×5 clip, clip cuối trùng mốc va chạm, quy đổi fps |
| `setup_env.sh`, `run_rq1b_train.sh` | Cài môi trường; train 3 seed (tự bỏ qua seed đã xong, tự chạy tiếp seed đang dở) |
| `infer_rq1b.py` | Chấm mọi epoch trên val, chọn checkpoint theo luật đã khóa, chấm official test |
| `analyze_rq1b.py` | Bảng và bootstrap CI trên val (dùng lại hàm của re03) và trên test (chạy nguyên re05) |

Toàn bộ quy trình đã được chạy thử trên CPU với dữ liệu giả: patch, tách frame, kiểm tra dữ liệu, train 1 epoch, chấm val và test, phân tích. Riêng `analyze_rq1b.py --stage val` đã được kiểm tra trên dữ liệu thật: nó tái lập đúng CI fixed − random của RQ3 (−0.0026 [−0.0311; 0.0249]).

## 5. Thời gian và chi phí (ước tính)

| Bước | Thời gian |
|---|---|
| Tải Nexar, cài môi trường, tách frame, cache val, kiểm tra | khoảng 1,5–2 giờ |
| Chạy thử 1 epoch | khoảng 10 phút |
| Train 3 seed | khoảng 8–10 giờ (A100). Con số chính xác lấy từ bước chạy thử. |
| Chấm val, chấm test, phân tích | khoảng 1 giờ |

**Chi phí:**
- A100 80 GB (khoảng $0,8–1,2/giờ): khoảng **$10–15**.
- GPU 48 GB với AMP (khoảng $0,4–0,6/giờ): khoảng **$6–8**.
