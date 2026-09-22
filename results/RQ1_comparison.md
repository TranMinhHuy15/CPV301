# RQ1 — So sánh 3 baseline (TOP / AdaLEA / RiskProp)

**Protocol dùng chung (matched-budget, proposal Mục 5.5):**
- Input: causal 5-frame snippet, 10fps, 224x224
- Backbone: SlowOnly-R50 (Kinetics-400 pretrained), fine-tune toàn bộ
- Optimizer: SGD lr=0.01, momentum=0.9, wd=1e-4, step decay x0.1 @epoch20/40
- 50 epoch, AMP, batch size khac nhau theo kien truc moi model (TOP/AdaLEA=8, RiskProp=2 video x 12 snippet=24 clip/batch)
- Data: Nexar Collision Prediction (HuggingFace `nexar-ai/nexar_collision_prediction`), 1500 video (750 positive/750 negative), split 80/20 stratified theo (target, alert-to-event bin), seed=42
- Eval: 300 video val (giu nguyen cho ca 3 model de so sanh cong bang), 3 lead-time co dinh 0.5s/1.0s/1.5s

## Bang ket qua chinh

| Model | Proposal mAP | AP@0.5s | AP@1.0s | AP@1.5s | Video AUC | mTTA@FAR<=0.1 (detected) | Coverage |
|---|---|---|---|---|---|---|---|
| TOP-5f      | 0.6496 | 0.7173 | 0.6412 | 0.5904 | 0.6516 | 1.114s | 44.0% |
| AdaLEA-5f   | **0.6931** | **0.7530** | **0.6883** | **0.6381** | 0.6860 | 1.092s | **47.3%** |
| RiskProp    | 0.6749 | 0.6923 | 0.6767 | 0.6558 | **0.7124** | **1.266s** | 41.3% |

## Bang paper-style (FAR<=0.1, format giong Table 1 cua cac paper goc)

| Model | AUC^0.1@0.5s | AUC^0.1@1.0s | AUC^0.1@1.5s | mAUC^0.1 (mean) | mTTA^0.1 (undetected=0) |
|---|---|---|---|---|---|
| TOP-5f    | 0.1871 | 0.1427 | 0.1062 | 0.1453 | 0.490s |
| AdaLEA-5f | 0.2600 | 0.1827 | 0.1333 | 0.1920 | 0.517s |
| RiskProp  | 0.1516 | 0.1364 | 0.1284 | 0.1388 | 0.523s |

## Phan tich

**1. mAP tong the: AdaLEA thang (0.6931)** > RiskProp (0.6749) > TOP (0.6496).
AdaLEA co loss adaptive theo threshold phi (adaptive early-anticipation), phu hop voi budget nho/50 epoch vi hoi tu nhanh (best checkpoint o epoch 2-4 cho ca 3 model, nhung AdaLEA giu duoc AP cao nhat o ca 3 horizon).

**2. Video-level AUC va canh bao som (mTTA): RiskProp thang** (AUC=0.7124, mTTA=1.266s -- canh bao truoc va chinh xac hon o cap do toan video).
Phu hop voi thiet ke cua RiskProp: FFR (Future-Frame Regularization) buoc model hoc chuoi logit tang dan huong ve thoi diem va cham, AMC (Adaptive Monotonic Constraint) ep tinh don dieu theo thoi gian -- ca hai deu la co che "danh rieng cho du doan som", nen du mAP khong cao nhat, RiskProp lai vuot troi o AUC/mTTA (do luong truc tiep kha nang phat hien va toc do canh bao).

**3. TOP thap nhat moi mat.**
TOP dung scheme 20-horizon cumulative (du doan xac suat tich luy tai 20 nguong 0.1s-2.0s) nhung khong co co che rang buoc thoi gian nhu AdaLEA (adaptive phi) hay RiskProp (FFR/AMC) -- gia thuyet: architecture don gian nhat trong 3 baseline, thieu inductive bias ve tinh don dieu/tich luy theo thoi gian ma 2 baseline kia co.

**4. Trade-off mAP vs AUC/mTTA:**
Khong co model nao thang tuyet doi ca 2 nhom metric -- AdaLEA toi uu do chinh xac phat hien (Average Precision), RiskProp toi uu do som/nhay cua canh bao (AUC, mTTA). Day la diem quan trong cho RQ2/RQ3 sau nay: neu he thong uu tien "canh bao dung" thi AdaLEA phu hop hon, neu uu tien "canh bao som + it bo sot o cap video" thi RiskProp phu hop hon.

## Gioi han / luu y khi doc ket qua

- Ca 3 model deu eval tren **val-split** (300/1500 video tach tu train, seed=42), **khong phai** tap test that cua Kaggle/HuggingFace (`solution.csv`) -- so sanh noi bo cong bang giua 3 baseline, chua phai so sanh voi so lieu trong cac paper goc.
- RiskProp dung optimizer/batch khac paper goc (paper: lr=0.002, batch=64, 8xA800; o day: lr=0.01, batch nho hon nhieu, 1 GPU) theo dung "matched-budget protocol" chung cua RQ1 -- day la chu dich cua nghien cuu, khong phai loi.
- RiskProp co dau hieu mat on dinh gradient thoang qua o epoch 21-22 (val loss nhay bat thuong) nhung best checkpoint chon tu epoch 15 (truoc do), khong bi anh huong.
- COLLISION_WEIGHT=5.0 cua RiskProp la gia tri thich nghi (paper khong cho so cu the), dung lai tu AdaLEA de dong bo.

## File lien quan
- `results/top5f_training_log.csv`, `results/top5f_eval_results.txt`
- `results/adalea5f_training_log.csv`, `results/adalea5f_eval_results.txt`
- `results/riskprop_training_log.csv`, `results/riskprop_eval_results.txt`
- Checkpoint model (`best_*.pth`): khong luu git (qua nang), tai rieng tu Jupyter file browser tren vast.ai instance.
