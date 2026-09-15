from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from mmtsfm.data.dataset import MMTSFMDataset

# Dataset-of-record backends (knowledge/protocol.md): disjoint cross-plant splits,
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
        history_days: float = 14.0,  # pv_record physical-time history (knowledge/protocol.md §3)
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
        visual_frame_spacing_min: float | None = None,  # None -> window / Tv
        # Ticket 45 burst sampling: set together to draw Tv frames as
        # Tv/frames_per_anchor dense bursts spaced anchor_stride_hours apart.
        visual_anchor_stride_hours: float | None = None,
        visual_frames_per_anchor: int | None = None,
        # A45a latent-depth control: keep only the NEWEST k cached latents, so a
        # shallower arm can reuse a deeper cache instead of re-extracting. Needs
        # video_frames == 2k (V-JEPA pools 2 frames per latent). None = load the
        # cache as extracted, which is what every pre-A45 run did.
        visual_latent_keep_newest: int | None = None,
        # A46b multi-anchor reuse: assemble Z from K cache entries at origins
        # t-(K-1)*visual_anchor_stride_hours ... t instead of one. Needs
        # video_frames == K * visual_frames_per_anchor. Turns a K-anchor arm
        # into a re-read of the existing 8-frame cache; raises on a miss rather
        # than falling back to a live encode of a different payload.
        visual_latent_anchors: int | None = None,
        vjepa_cache_dir: Optional[str] = None,
        emit_vision: bool = True,  # False for vision-free runs (skip frame decode + latents)
        # A36 / ticket 40 — what the covariate block carries over the HORIZON.
        # "all" (default, and what every recorded MMTSFM number was produced
        # under) exposes the full 14-covariate protocol block as known future,
        # NWP-style irradiance included. "deterministic" zeroes everything that
        # is not knowable in advance and keeps only solar geometry / calendar /
        # clearsky GHI (`baselines/common/config.py::DETERMINISTIC_COVS`), which
        # is the history-only regime ticket 40 asks about. Validated downstream
        # by `WindowDataset`; ignored by the synthetic dataset.
        future_cov: str = "all",
        # pv_record TRAIN window stride. Default None → stride 1 (every step is a
        # window origin). uk_pv stride-1 ≈ 1.36M train windows — set >1 to bound
        # epoch size AND the pre-extracted V-JEPA cache (extractor must use the
        # SAME value or cache keys miss). val/test always stride=H (protocol).
        train_stride: Optional[int] = None,
        # num_samples_* only used by "synthetic"; real datasets compute their own length
        num_samples_train: int = 1000,
        num_samples_val: int = 200,
        num_samples_test: int = 200,
        train_frac: float = 0.70,
        val_frac: float = 0.10,
        vis_cadence_multiplier: int = 1,
        # Test order. False (protocol default) = the deterministic series-major
        # order every recorded number was produced under. The ONE caller that
        # needs True is the A10 mismatched-plant control: test windows are laid
        # out series-major (~984 per plant at stride=H), so an ordered batch is
        # a single site and no sample can be handed a DIFFERENT plant's sky.
        # Metrics are aggregated per plant from site_id and are order-invariant,
        # so this changes which batch the attention diagnostic captures and
        # nothing else.
        shuffle_test: bool = False,
        shuffle_test_seed: int = 42,
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
                visual_frame_spacing_min=self.hparams.visual_frame_spacing_min,
                visual_anchor_stride_hours=self.hparams.visual_anchor_stride_hours,
                visual_frames_per_anchor=self.hparams.visual_frames_per_anchor,
                visual_latent_keep_newest=self.hparams.visual_latent_keep_newest,
                visual_latent_anchors=self.hparams.visual_latent_anchors,
                # W4: cross-plant mixing is a TRAIN-time mechanism. val/test keep
                # N=1 so per-plant protocol metrics + site_id collate are unchanged.
                num_entities=self.hparams.num_entities if split == "train" else 1,
                h5_path=self.hparams.h5_path,
                vjepa_cache_dir=self.hparams.vjepa_cache_dir,
                emit_vision=self.hparams.emit_vision,
                future_cov=self.hparams.future_cov,
                stride=self.hparams.train_stride if split == "train" else None,
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
            vjepa_cache_dir=self.hparams.vjepa_cache_dir,
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
        self,
        dataset: MMTSFMDataset,
        shuffle: bool,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
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
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if not self.hparams.shuffle_test:
            return self._loader(self.test_dataset, shuffle=False)
        # Seeded so the shuffled control is reproducible run to run.
        gen = torch.Generator()
        gen.manual_seed(int(self.hparams.shuffle_test_seed))
        return self._loader(self.test_dataset, shuffle=True, generator=gen)
