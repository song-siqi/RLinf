# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import tqdm
import tyro
from openpi.training.config import DataConfig

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def _split_repo_ids(repo_id: str) -> list[str]:
    return [item.strip() for item in repo_id.split(",") if item.strip()]


def _read_episode_indices(dataset_root: Path) -> list[int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    indices = []
    with episodes_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                indices.append(int(json.loads(line)["episode_index"]))
    return indices


def compute_local_lerobot_stats(
    repo_id: str, keys: list[str], target_dim: int | None = None
):
    """Compute state/action stats directly from local parquet files.

    Norm stats do not need images, so avoid instantiating LeRobotDataset here:
    its __getitem__ decodes videos and can fail when FFmpeg/torchcodec is not
    available in the environment.
    """
    import pyarrow.parquet as pq

    hf_lerobot_home = Path(os.environ["HF_LEROBOT_HOME"])
    repo_ids = _split_repo_ids(repo_id)
    stats = {key: normalize.RunningStats() for key in keys}
    found_any = False

    for dataset_name in repo_ids:
        dataset_root = hf_lerobot_home / dataset_name
        info_path = dataset_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"LeRobot metadata not found: {info_path}. "
                "Check HF_LEROBOT_HOME and --repo-id."
            )

        with info_path.open() as f:
            info = json.load(f)
        missing = [key for key in keys if key not in info.get("features", {})]
        if missing:
            raise KeyError(
                f"Dataset {dataset_name!r} is missing required features: {missing}."
            )

        data_path_template = info["data_path"]
        chunks_size = int(info.get("chunks_size", 1000))
        episode_indices = _read_episode_indices(dataset_root)

        for episode_index in tqdm.tqdm(
            episode_indices, desc=f"Reading {dataset_name}", leave=False
        ):
            episode_chunk = int(episode_index) // chunks_size
            parquet_path = dataset_root / data_path_template.format(
                episode_chunk=episode_chunk,
                episode_index=int(episode_index),
            )
            table = pq.read_table(parquet_path, columns=keys)
            for key in keys:
                values = np.asarray(table[key].to_pylist(), dtype=np.float32)
                if target_dim is not None and key in {"state", "actions"}:
                    values = transforms.pad_to_dim(values, target_dim)
                stats[key].update(values)
            found_any = True

    if not found_any:
        raise ValueError(f"No frames found for repo_id={repo_id!r}.")
    return {key: stats.get_statistics() for key, stats in stats.items()}


def _stats_output_path(config, data_config: DataConfig) -> Path:
    assets_dir = config.data.assets.assets_dir or config.assets_dirs
    return Path(assets_dir) / (data_config.asset_id or data_config.repo_id)


def create_torch_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.TorchDataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    repo_id: str,
):
    if not os.environ.get("HF_LEROBOT_HOME"):
        raise EnvironmentError(
            "HF_LEROBOT_HOME must be set before running this script. "
            "Export it manually, for example: "
            "export HF_LEROBOT_HOME=/path/to/lerobot_root"
        )
    config = get_openpi_config(
        config_name,
        data_kwargs={"repo_id": repo_id},
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    keys = ["state", "actions"]
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size
        )
        stats = {key: normalize.RunningStats() for key in keys}
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    else:
        norm_stats = compute_local_lerobot_stats(
            data_config.repo_id, keys, target_dim=config.model.action_dim
        )

    output_path = _stats_output_path(config, data_config)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
