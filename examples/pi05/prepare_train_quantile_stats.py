#!/usr/bin/env python
"""Add train-split q01/q99 to a local v3 dataset's stats.json for PI0.5."""

import argparse
from copy import deepcopy
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.augment_dataset_quantile_stats import collect_episode_arrays
from lerobot.utils.constants import ACTION, OBS_STATE


def merge_quantile_stats(
    existing_stats: dict[str, dict],
    quantile_stats: dict[str, dict[str, object]],
) -> dict[str, dict]:
    """Copy existing feature stats and replace only q01/q99 with new values."""
    merged = deepcopy(existing_stats)
    for key, values in quantile_stats.items():
        if key not in merged:
            raise KeyError(f"Cannot add quantiles for missing dataset statistics feature {key!r}")
        for quantile in ("q01", "q99"):
            merged[key][quantile] = values[quantile].tolist()
    return merged


def prepare_train_quantile_stats(
    repo_id: str,
    source_root: Path,
    eval_split: float,
) -> dict[str, object]:
    """Compute q01/q99 on train episodes and merge them into local metadata."""
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=repo_id,
            root=str(source_root),
            eval_split=eval_split,
            use_imagenet_stats=False,
        ),
        policy=PI05Config(),
    )
    train_dataset, _ = make_train_eval_datasets(cfg)
    train_episodes = train_dataset.episodes or list(range(train_dataset.num_episodes))

    accumulators = {key: RunningQuantileStats() for key in (OBS_STATE, ACTION)}
    for episode in train_episodes:
        arrays = collect_episode_arrays(train_dataset, episode, use_sampling=False, skip_images=True)
        for key, accumulator in accumulators.items():
            if key not in arrays:
                raise KeyError(f"Dataset episode {episode} has no numeric feature {key!r}")
            accumulator.update(arrays[key][0])

    computed_quantiles = {key: accumulator.get_statistics() for key, accumulator in accumulators.items()}
    original_stats = load_stats(source_root)
    if original_stats is None:
        raise FileNotFoundError(source_root / "meta" / "stats.json")
    merged_stats = merge_quantile_stats(original_stats, computed_quantiles)
    write_stats(merged_stats, source_root)
    return {"train_episodes": len(train_episodes), "stats": merged_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--eval-split", type=float, required=True, help="Same dataset.eval_split used for training")
    args = parser.parse_args()
    result = prepare_train_quantile_stats(args.repo_id, args.source_root, args.eval_split)
    state_q01 = result["stats"][OBS_STATE]["q01"]
    action_q01 = result["stats"][ACTION]["q01"]
    print(
        f"Prepared local quantile metadata for {result['train_episodes']} training episodes; "
        f"state/action q01 dims={len(state_q01)}/{len(action_q01)}; "
        f"stats={args.source_root / 'meta' / 'stats.json'}"
    )


if __name__ == "__main__":
    main()
