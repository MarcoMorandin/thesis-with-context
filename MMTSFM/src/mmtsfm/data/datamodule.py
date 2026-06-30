from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from mmtsfm.data.dataset import MMTSFMDataset

# Dataset-of-record backends (BASELINE_PROTOCOL.md): disjoint cross-plant splits,
# physical-time windows, protocol covariates incl. known future weather + frames.
PV_RECORD_DATASETS = ("uk_pv", "goes_pvdaq")


def _collate_optional_z(batch):
    """Default collate that gracefully handles an optional 'Z' key.

    If ALL items have 'Z', stack it.  If NONE do, omit it.
    Mixed batches (partial cache hit) drop 'Z' from all items so
    the batch shape is always consistent.
    """
    has_z = ["Z" in b for b in batch]
    if not all(has_z):
        batch = [{k: v for k, v in b.items() if k != "Z"} for b in batch]
    return torch.utils.data.dataloader.default_collate(batch)


class MMTSFMDataModule(LightningDataModule):
    """LightningDataModule for MMTSFMDataset.

    Supports all dataset_name values accepted by MMTSFMDataset.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        dataset_name: str = "synthetic",
        batch_size: int = 16,
        num_workers: int = 0,
        num_entities: int = 10,
        hist_steps: int = 24,
        horizon: int = 12,
        history_days: float = 14.0,  # pv_record physical-time history (BASELINE_PROTOCOL §3)
        horizon_hours: float = 6.0,  # pv_record physical-time horizon
        h5_path: Optional[
            str
        ] = None,  # pv_record frames; default <data_dir>/images_all.h5
        target_dim: int = 1,
        covariate_dim: int = 5,
        video_frames: int = 8,
        img_channels: int = 3,
        img_size: int = 64,
        imagenet_norm: bool = False,
        visual_window_hours: float = 6.0,  # W5: recency cap on candidate frames
        vidtok_cache_dir: Optional[str] = None,
        # num_samples_* only used by "synthetic"; real datasets compute their own length
        num_samples_train: int = 1000,
        num_samples_val: int = 200,
        num_samples_test: int = 200,
        train_frac: float = 0.70,
        val_frac: float = 0.10,
        vis_cadence_multiplier: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _make_dataset(self, split: str, num_samples: int):
        if self.hparams.dataset_name in PV_RECORD_DATASETS:
            from mmtsfm.data.pv_record import PVRecordDataset

            return PVRecordDataset(
                split=split,
                dataset_name=self.hparams.dataset_name,
                data_path=self.hparams.data_dir,
                history_days=self.hparams.history_days,
                horizon_hours=self.hparams.horizon_hours,
                hist_steps=self.hparams.hist_steps or None,
                horizon=self.hparams.horizon or None,
                video_frames=self.hparams.video_frames,
                img_size=self.hparams.img_size,
                img_channels=self.hparams.img_channels,
                imagenet_norm=self.hparams.imagenet_norm,
                visual_window_hours=self.hparams.visual_window_hours,
                # W4: cross-plant mixing is a TRAIN-time mechanism. val/test keep
                # N=1 so per-plant protocol metrics + site_id collate are unchanged.
                num_entities=self.hparams.num_entities if split == "train" else 1,
                h5_path=self.hparams.h5_path,
            )
        return MMTSFMDataset(
            num_samples=num_samples,
            data_dir=self.hparams.data_dir,
            dataset_name=self.hparams.dataset_name,
            split=split,
            num_entities=self.hparams.num_entities,
            hist_steps=self.hparams.hist_steps,
            horizon=self.hparams.horizon,
            target_dim=self.hparams.target_dim,
            covariate_dim=self.hparams.covariate_dim,
            video_frames=self.hparams.video_frames,
            img_channels=self.hparams.img_channels,
            img_size=self.hparams.img_size,
            imagenet_norm=self.hparams.imagenet_norm,
            vidtok_cache_dir=self.hparams.vidtok_cache_dir,
            train_frac=self.hparams.train_frac,
            val_frac=self.hparams.val_frac,
            vis_cadence_multiplier=self.hparams.vis_cadence_multiplier,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = self._make_dataset(
                "train", self.hparams.num_samples_train
            )
            self.val_dataset = self._make_dataset("val", self.hparams.num_samples_val)

        if stage in ("test", None):
            self.test_dataset = self._make_dataset(
                "test", self.hparams.num_samples_test
            )

    def _loader(
        self, dataset: MMTSFMDataset, shuffle: bool, drop_last: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=_collate_optional_z,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
