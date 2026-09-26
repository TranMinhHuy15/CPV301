"""
RQ1b -- Nexar dataset for the authors' RiskProp code, with two documented
changes. Imported through `custom_imports` in configs/riskprop_full_nexar.py.

This subclasses the authors' taa.datasets.MultiDataset and replaces ONLY its
Nexar branch of load_data_list(). The code below is the authors' Nexar branch
(taa/datasets.py, commit 579376f) with two changes:

  (a) fps = 10 instead of the hard-coded 30. Frames are extracted at 10 fps by
      extract_frames_10fps.py, and annotations.csv stores frame numbers on the
      same 10 fps grid (frame k = time k/10 s; accident_frame =
      round(time_of_event * 10), abnormal_start_frame = round(time_of_alert *
      10)). With fps = 10 the authors' sampler uses frame_interval = 1, so a
      clip is 5 consecutive 10-fps frames and consecutive clips are 0.1 s apart
      -- the same time scale as the authors' 30 fps / frame_interval 3 setup.
  (b) The validation split is the project's 300-video split
      (results/repro/split_manifest_seed42.json, "val_ids") instead of the
      authors' `nexar_val` list in taa/splits.py. The other 1,200 videos are
      the training set, exactly as in RQ1.

Everything else (which videos are kept in train / test mode, field names, the
is_val flag used by the authors' metric) is unchanged.
"""
import json
import os.path as osp

import pandas as pd
from mmaction.registry import DATASETS
from taa.datasets import MultiDataset


def load_val_ids(split_manifest):
    with open(split_manifest) as f:
        man = json.load(f)
    ids = {str(int(v)).zfill(5) for v in man["val_ids"]}
    if len(ids) != 300:
        raise ValueError(f"expected 300 val ids in {split_manifest}, got {len(ids)}")
    return ids


@DATASETS.register_module()
class NexarRQ1BDataset(MultiDataset):
    def __init__(self, split_manifest, fps=10, **kwargs):
        # must be set before MultiDataset.__init__ -> full_init -> load_data_list
        self.split_manifest = split_manifest
        self.rq1b_fps = int(fps)
        self.rq1b_val_ids = load_val_ids(split_manifest)
        for k in ("cap", "dada", "d2city"):
            if kwargs.get(k):
                raise ValueError(f"RQ1b uses Nexar only; got {k}={kwargs[k]!r}")
        super().__init__(**kwargs)

    def load_data_list(self):
        if not self.nexar:
            raise ValueError("NexarRQ1BDataset needs nexar=dict(data_root=..., ann_file=...)")
        nexar_val = self.rq1b_val_ids          # change (b)
        data_list = []
        fin = pd.read_csv(osp.join(self.nexar["data_root"], self.nexar["ann_file"])).values.tolist()
        for line in fin:
            video_id = str(int(line[0])).zfill(5)
            is_test = bool(line[1])
            target = bool(line[6]) if not is_test else None
            if not is_test:
                filename = "train"
                frame_dir = "train_raw_frames"
            else:
                filename = "test"
                frame_dir = "test_raw_frames"
            filename = osp.join(self.nexar["data_root"], filename, video_id + ".mp4")
            frame_dir = osp.join(self.nexar["data_root"], frame_dir, video_id)
            fps = self.rq1b_fps                 # change (a): was `fps = 30`

            # keep the train videos
            if not self.test_mode and is_test:
                continue

            if not self.test_mode and not self.train_with_val and video_id in nexar_val:
                continue

            # keep the test videos
            if self.test_mode and not self.val_train and video_id not in nexar_val and not is_test:
                continue

            data_list.append(
                dict(
                    dataset="nexar",
                    filename=filename,
                    frame_dir=frame_dir,
                    filename_tmpl=self.nexar["filename_tmpl"],
                    start_index=self.nexar["start_index"],
                    video_id=video_id,
                    type=None,
                    target=target,
                    abnormal_start_frame=int(line[4]) if not is_test and target else None,
                    accident_frame=int(line[5]) if not is_test and target else None,
                    total_frames=int(line[2]),
                    fps=fps,
                    is_val=video_id in nexar_val,
                    is_test=is_test,
                )
            )
        return data_list
