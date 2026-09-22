# Process Log -- RQ1 Baseline Training (TOP / AdaLEA / RiskProp)

Ngay chay: 2026-09-22
Moi truong: vast.ai, GPU 1x RTX 4070 Ti SUPER (16GB VRAM, 42 TFLOPS), gia thue $0.203/hr, truy cap qua Jupyter Terminal (khong dung SSH key)
Dataset: HuggingFace `nexar-ai/nexar_collision_prediction` (31.4GB, 1500 video train, 750 positive/750 negative)

## 1. Setup moi truong (~15 phut)

| Buoc | Lenh chinh | Ghi chu |
|---|---|---|
| Clone code | `git clone https://github.com/TranMinhHuy15/CPV301.git` | Repo flat structure (14 file .py goc, khong co subfolder do loi upload GitHub UI truoc do) |
| Cai thu vien | `pip install decord pytorchvideo scikit-learn pandas pillow huggingface_hub` | |
| Tai dataset | `hf download nexar-ai/nexar_collision_prediction --repo-type dataset --local-dir /workspace/data/nexar_hf` | ~24 phut tai (unauthenticated, ~20-39MB/s), 31.4GB |
| Build train.csv | Convert `train/positive/metadata.csv` + `train/negative/metadata.csv` (HuggingFace videofolder format) -> `train.csv` format Kaggle cu (id,target,time_of_event,time_of_alert) | Video flatten qua symlink vao 1 thu muc |
| Fix path | `sed` doi `/kaggle/working` -> `/workspace/CPV301`, `/kaggle/input/...` -> `/workspace/data/nexar` trong toan bo file .py | |
| Fix thieu file | Tai tao `cell11b_cached_dataset.py` + `cell13_model.py` (TOP model/dataset -- 2 file da chay tren Kaggle truoc do nhung chua tung duoc luu ra .py rieng nen thieu trong repo GitHub) | Xac dinh dung format 20-horizon cumulative label (0.1s-2.0s, bin 0.1s) tu doi chieu voi `cell16_eval_cached_5f.py` (LEAD_TO_HIDX mapping) |

## 2. TOP-5f

| Buoc | Script | Thoi gian |
|---|---|---|
| Precache (5-frame causal window, 3 window/video train + 3 lead-time/video val) | `cell10_precache_5f.py` | 8.0 phut (cache 3.2GB) |
| Train (50 epoch, SGD lr=0.01, step decay @20/40, pos_weight=10, AMP) | `cell15_train_cached_5f.py` | 7.7 phut (best checkpoint epoch 4) |
| Eval (3 lead-time 0.5/1.0/1.5s, FAR<=0.1) | `cell16_eval_cached_5f.py` | ~1 phut |
| **Ket qua** | Proposal mAP=0.6496, Video AUC=0.6516, mTTA=1.114s | |

## 3. AdaLEA-5f (tai dung cache 5f cua TOP, khong precache rieng)

| Buoc | Script | Thoi gian |
|---|---|---|
| Train (50 epoch, adaptive threshold phi) | `cell23_train_adalea_cached.py` | 11.9 phut (best checkpoint epoch 2) |
| Eval | `cell24_eval_adalea_cached.py` | ~1 phut |
| **Ket qua** | Proposal mAP=0.6931, Video AUC=0.6860, mTTA=1.092s | |

## 4. RiskProp (can precache rieng -- sequence cache moi)

| Buoc | Script | Thoi gian |
|---|---|---|
| Precache (sequence 12 snippet/video, stride 0.5s, chi train; val tai dung nexar_cache_5f/val) | `cell30_precache_riskprop.py` | 12.2 phut (cache 11GB, 1200 video x 12 snippet) |
| Train (50 epoch, FFR+AMC loss, lambda1=1.5, lambda2=1.1) | `cell34_train_riskprop_cached.py` | 81.9 phut (best checkpoint epoch 15) -- nang nhat vi 24 clip/batch thay vi 8 |
| Eval | `cell35_eval_riskprop_cached.py` | ~1 phut |
| **Ket qua** | Proposal mAP=0.6749, Video AUC=0.7124, mTTA=1.266s | |

## 5. Tong thoi gian + chi phi

- Tong thoi gian chay (setup + 3 baseline): ~2.5 tieng lien tuc
- Chi phi GPU: ~2.5h x $0.203/hr ≈ **$0.51** (trong ngan sach $5 credit)
- Van con chay tren instance (chua Destroy) de con dung tiep neu can

## 6. Van de gap phai + cach xu ly

1. **Repo GitHub upload flat (khong co subfolder top/adalea/riskprop)** -- do loi keo-tha khi upload qua GitHub web UI -- chap nhan giu flat vi ten file khong trung nhau.
2. **Thieu 2 file `cell11b_cached_dataset.py` + `cell13_model.py`** (TOP) -- da chay thanh cong tren Kaggle notebook truoc do nhung chua luu ra file rieng -- tai tao lai dua tren doi chieu cau truc `cell16_eval_cached_5f.py` (LEAD_TO_HIDX) + pattern cua AdaLEA/RiskProp da co san.
3. **RiskProp mat on dinh gradient thoang qua o epoch 21-22** (val loss nhay len 6.45, bat thuong) -- khong anh huong vi best checkpoint da chon truoc do (epoch 15).
4. **GitHub push can Personal Access Token** (khong dung password thuong duoc nua) -- tao token classic voi scope `repo`.

## 7. Con thieu / buoc tiep theo (chua lam hom nay)

- Chua eval tren tap test that cua Nexar (`solution.csv`, `time_to_accident_test_map.csv` tu HuggingFace) -- hien tai ca 3 model dang eval tren val-split (300/1500 video tach tu train) de so sanh noi bo cong bang, chua phai so sanh literature-comparable.
- RQ2 (RiskProp 4-way ablation: neither/FFR-only/AMC-only/both) -- chua bat dau.
- RQ3 (fixed-lag vs random-offset AMC pairing) -- chua bat dau.
- Checkpoint (`best_*.pth`) chua tai ve may (van con tren vast.ai instance) -- can tai qua Jupyter file browser truoc khi Destroy.

## File ket qua lien quan (trong `results/`)
- `RQ1_comparison.md` -- bang so sanh + phan tich chi tiet
- `top5f_*`, `adalea5f_*`, `riskprop_*` -- log training (csv) + ket qua eval (txt) tung baseline
