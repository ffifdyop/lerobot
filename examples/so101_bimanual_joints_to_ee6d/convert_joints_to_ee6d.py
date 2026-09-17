# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert a bimanual SO-101 joint-space LeRobot dataset (v3.0) to end-effector 6D pose space.

For every frame, forward kinematics (placo, via `RobotKinematics`) maps the 6 joints of each
arm to an EE pose. `action` and `observation.state` are replaced by 14-dim vectors:

    [left_ee.x, left_ee.y, left_ee.z, left_ee.wx, left_ee.wy, left_ee.wz, left_ee.gripper_pos,
     right_ee.x, ..., right_ee.gripper_pos]

- position (x, y, z) in meters, in each arm's own base frame
- orientation (wx, wy, wz) as rotation vector (axis * angle), in radians
- gripper_pos: raw gripper joint value, unchanged (degrees)

Videos are hard-linked (not re-encoded); parquet data, meta/info.json, meta/stats.json and
per-episode stats in meta/episodes are rewritten. The source dataset is left untouched.

Run from the repository root:

    uv run python examples/so101_bimanual_joints_to_ee6d/convert_joints_to_ee6d.py
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from lerobot.model.kinematics import RobotKinematics

SRC = Path.home() / ".cache/huggingface/lerobot/ffifdyop/so101_bimanual_folding-cloth-720p"
DST = Path.home() / ".cache/huggingface/lerobot/ffifdyop/so101_bimanual_folding-cloth-720p_ee6d"

URDF_PATH = "SO101/so101_new_calib.urdf"
TARGET_FRAME = "gripper_frame_link"
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

CONVERTED_FEATURES = ["action", "observation.state"]
EE_COMPONENTS = ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]
EE_NAMES = [f"{side}_ee.{comp}" for side in ("left", "right") for comp in EE_COMPONENTS]


def make_kinematics() -> RobotKinematics:
    return RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=ARM_JOINT_NAMES,
    )


def joints_to_ee6d(kinematics: RobotKinematics, joints_deg: np.ndarray) -> list[float]:
    """6 arm joints (degrees) -> [x, y, z, wx, wy, wz, gripper_pos]."""
    t = kinematics.forward_kinematics(joints_deg)
    pos = t[:3, 3]
    rotvec = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(rotvec[0]), float(rotvec[1]), float(rotvec[2]), float(joints_deg[5])]


def convert_column(kinematics: RobotKinematics, values: np.ndarray) -> np.ndarray:
    """(N, 12) joint matrix -> (N, 14) bimanual ee6d matrix."""
    out = np.empty((values.shape[0], 14), dtype=np.float32)
    for i, row in enumerate(values):
        out[i, :7] = joints_to_ee6d(kinematics, row[:6])
        out[i, 7:] = joints_to_ee6d(kinematics, row[6:12])
    return out


def column_stats(values: np.ndarray) -> dict[str, list]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [values.shape[0]],
    }


def copy_dataset(src: Path, dst: Path) -> None:
    """Copy everything except videos/ (real copies); hard-link video files."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("videos"))
    for path in sorted((src / "videos").rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def rewrite_data_parquet(path: Path, converted: dict[str, np.ndarray]) -> None:
    table = pq.read_table(path)
    for feature, values in converted.items():
        col_index = table.schema.get_field_index(feature)
        lists = pa.array([row.tolist() for row in values], type=pa.list_(pa.float32()))
        table = table.set_column(col_index, feature, lists)
    path.unlink()  # may be a hard link sibling; ensure we never touch the source
    pq.write_table(table, path)


def main() -> None:
    kinematics = make_kinematics()
    copy_dataset(SRC, DST)

    info_path = DST / "meta/info.json"
    info = json.loads(info_path.read_text())

    # Convert every data parquet and collect converted arrays for stats.
    all_converted: dict[str, list[np.ndarray]] = {f: [] for f in CONVERTED_FEATURES}
    all_episode_indices: list[np.ndarray] = []
    data_files = sorted(DST.glob("data/chunk-*/file-*.parquet"))
    for data_path in data_files:
        table = pq.read_table(data_path)
        episode_indices = table.column("episode_index").to_numpy()
        converted = {}
        for feature in CONVERTED_FEATURES:
            joints = np.stack(table.column(feature).to_pylist()).astype(np.float64)
            converted[feature] = convert_column(kinematics, joints)
            all_converted[feature].append(converted[feature])
        all_episode_indices.append(episode_indices)
        rewrite_data_parquet(data_path, converted)
        print(f"converted {data_path.relative_to(DST)} ({table.num_rows} frames)")

    episode_indices = np.concatenate(all_episode_indices)

    # Global stats.
    stats_path = DST / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    for feature in CONVERTED_FEATURES:
        values = np.concatenate(all_converted[feature])
        stats[feature] = column_stats(values.astype(np.float64))
    stats_path.write_text(json.dumps(stats, indent=4))

    # Per-episode stats embedded in meta/episodes parquet.
    for episodes_path in sorted(DST.glob("meta/episodes/chunk-*/file-*.parquet")):
        table = pq.read_table(episodes_path)
        ep_ids = table.column("episode_index").to_numpy()
        for feature in CONVERTED_FEATURES:
            values = np.concatenate(all_converted[feature])
            for stat_name in ("min", "max", "mean", "std", "count"):
                per_episode = []
                for ep in ep_ids:
                    ep_stats = column_stats(values[episode_indices == ep].astype(np.float64))
                    per_episode.append(ep_stats[stat_name])
                col_name = f"stats/{feature}/{stat_name}"
                col_index = table.schema.get_field_index(col_name)
                dtype = pa.list_(pa.int64()) if stat_name == "count" else pa.list_(pa.float64())
                table = table.set_column(col_index, col_name, pa.array(per_episode, type=dtype))
        episodes_path.unlink()
        pq.write_table(table, episodes_path)
        print(f"updated {episodes_path.relative_to(DST)}")

    # Feature schema: same 14 ee6d names for action and observation.state.
    for feature in CONVERTED_FEATURES:
        info["features"][feature]["shape"] = [14]
        info["features"][feature]["names"] = EE_NAMES
    info_path.write_text(json.dumps(info, indent=4))

    print(f"done -> {DST}")


if __name__ == "__main__":
    main()
