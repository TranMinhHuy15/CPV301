"""
Tai nexar-ai/nexar_collision_prediction tu HuggingFace ve va dung lai
thanh dung format Kaggle (train.csv + train/{id}.mp4) de cell10/cell30
chay khong can sua gi them.

Chay 1 lan DUY NHAT truoc cell10/cell30.
Can: pip install datasets huggingface_hub decord --break-system-packages
Neu repo yeu cau dang nhap: hf auth login (token co quyen doc dataset nay)

v2 (2026-09-24): ban "datasets" moi hon khong con field 'label' rieng va
Video feature mac dinh decode=True (tra ve VideoDecoder thay vi path) ->
sua lai: tat decode video, va suy ra target tu duong dan file
(.../train/positive/xxxxx.mp4 hoac .../train/negative/xxxxx.mp4).
"""
import os
import pandas as pd
from huggingface_hub import snapshot_download
from datasets import load_dataset
from datasets import Video

HF_REPO       = "nexar-ai/nexar_collision_prediction"
HF_LOCAL_RAW  = "/workspace/CPV301/data/nexar_hf_raw"
OUT_DATA_DIR  = "/workspace/CPV301/data/nexar_kaggle_style"
OUT_TRAIN_DIR = os.path.join(OUT_DATA_DIR, "train")

os.makedirs(OUT_TRAIN_DIR, exist_ok=True)

print(f"Downloading {HF_REPO} (chi lay train/, bo qua test-public/test-private)...")
snapshot_download(
    repo_id=HF_REPO, repo_type="dataset", local_dir=HF_LOCAL_RAW,
    allow_patterns=["train/**"],
)

print("Load qua datasets 'videofolder' builder (dung code mau trong README HF)...")
train_dir = os.path.join(HF_LOCAL_RAW, "train")
ds = load_dataset("videofolder", data_dir=train_dir, split="train", drop_labels=False)

# Tat decode tu dong cua video: giu nguyen dict {'path':..., 'bytes':...}
# thay vi tra ve VideoDecoder da giai ma (ban datasets moi doi hanh vi mac dinh).
if "video" in ds.features and isinstance(ds.features["video"], Video):
    ds = ds.cast_column("video", Video(decode=False))

# --- KIEM TRA TRUOC KHI CHAY HET: in field va 1 vi du de xac nhan dung cot ---
print("Cac field co san:", ds.features)
ex0 = ds[0]
print("Vi du dau tien:", {k: v for k, v in ex0.items() if k != "video"})
print("Video path vi du:", ex0["video"]["path"] if isinstance(ex0["video"], dict) else ex0["video"])
print("(target se duoc suy ra tu duong dan chua '/positive/' hoac '/negative/', "
      "vi field 'label' khong con trong dataset)")
input("Neu thong tin tren hop ly, nhan Enter de tiep tuc xu ly toan bo. "
      "Neu KHONG dung, Ctrl+C va bao lai.")


def get_video_path(ex):
    v = ex["video"]
    return v["path"] if isinstance(v, dict) else v


def infer_target(ex, path):
    # Uu tien field 'label' neu ton tai va khac None (tuong thich ban dataset cu)
    label = ex.get("label")
    if label is not None:
        return int(label)
    # Nguoc lai suy tu duong dan thu muc con (positive/ vs negative/)
    p = (path or "").replace("\\", "/").lower()
    if "/positive/" in p:
        return 1
    if "/negative/" in p:
        return 0
    raise ValueError(f"Khong suy ra duoc target tu duong dan: {path}")


rows = []
for i, ex in enumerate(ds):
    vid_id = f"{i:05d}"
    dst_path = os.path.join(OUT_TRAIN_DIR, f"{vid_id}.mp4")
    src_path = get_video_path(ex)
    target = infer_target(ex, src_path)
    if not os.path.exists(dst_path):
        os.symlink(os.path.abspath(src_path), dst_path)
    rows.append({
        "id": int(vid_id),
        "target": target,
        "time_of_event": ex.get("time_of_event"),
        "time_of_alert": ex.get("time_of_alert"),
    })
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(ds)} videos indexed...")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DATA_DIR, "train.csv"), index=False)
print(f"\nXong: {len(df)} videos -> {OUT_DATA_DIR}/train.csv + {OUT_TRAIN_DIR}/*.mp4")
print(f"target=1: {df['target'].sum()} | target=0: {(df['target']==0).sum()}")
