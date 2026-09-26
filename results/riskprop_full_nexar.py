# RQ1b -- RiskProp-full on Nexar.
#
# Inherits the authors' snippet-level config UNCHANGED (github.com/xingyueye5/RiskProp,
# commit 579376f, configs/predict_anomaly_snippet.py + _base_/schedules/sgd_50e.py):
#   SlowOnly-R50 (Kinetics-710 weights via load_from), 30 clips x 5 frames, clips 1 frame
#   (0.1 s) apart, RandomResizedCrop + flip, AnticipationHead with label_with="constraint"
#   (BCE + 1.5 FFR + 1.1 AMC, 30 pairs), SGD lr 0.002 / momentum 0.9 / wd 1e-4,
#   grad clip 40, MultiStepLR at epochs 20 and 40, 50 epochs, batch 2 videos per GPU.
#
# Changes (all documented as deviations in RESULTS_RQ1B.md):
#   1. Dataset: Nexar only (the public config trains on CAP; its Nexar lines are commented
#      out), through NexarRQ1BDataset = the authors' Nexar loader with fps = 10 and the
#      project's 300-video validation split (riskprop_full/nexar_rq1b.py).
#   2. One GPU instead of eight: gradient accumulation over 8 iterations (2 x 8 = 16 videos
#      per optimizer step, as in the README's 8-GPU command; note the repo's own
#      dist_train.sh uses 6 GPUs = 12). BatchNorm still sees 2 videos x 30 clips per forward,
#      as on each of the authors' GPUs.
#   3. Checkpoints of every epoch are kept, so the project's locked checkpoint rule can be
#      applied afterwards (riskprop_full/infer_rq1b.py). The authors' own rule (best mAUC@
#      on validation) is kept too, for the appendix.
#
# This file must sit next to predict_anomaly_snippet.py (copy it into <RiskProp>/configs/)
# and training must be started from the RiskProp repo root, as in the authors' scripts.

_base_ = ["predict_anomaly_snippet.py"]

custom_imports = dict(imports=["taa", "nexar_rq1b"], allow_failed_imports=False)

split_manifest = "/workspace/CPV301/results/repro/split_manifest_seed42.json"
nexar = dict(
    data_root="data/nexar-collision-prediction",
    ann_file="annotations.csv",
    filename_tmpl="{:06}.jpg",
    start_index=0,
)

train_dataloader = dict(
    dataset=dict(type="NexarRQ1BDataset", split_manifest=split_manifest, fps=10, cap=None, nexar=nexar)
)
val_dataloader = dict(
    dataset=dict(type="NexarRQ1BDataset", split_manifest=split_manifest, fps=10, cap=None, nexar=nexar)
)
test_dataloader = val_dataloader

optim_wrapper = dict(accumulative_counts=8)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=50, save_best="mAUC@", rule="greater")
)

randomness = dict(seed=42, deterministic=False)
