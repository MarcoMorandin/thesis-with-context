"""Protocol windows for the library arm — MMTSFM's dataset, plain DataLoaders.

``PVRecordDataset`` *is* the MMTSFM training set (committed disjoint plant
splits, physical-time windows, protocol covariates with known future weather).
It is a plain ``torch.utils.data.Dataset``, so it is used here directly instead
of ``MMTSFMDataModule``: that datamodule is a ``lightning.pytorch`` object,
while ``neuralforecast`` is built on the separate ``pytorch_lightning``
distribution, and mixing the two inside one module tree is not supported.
Loader semantics below are copied from ``MMTSFMDataModule._loader``.
"""

from __future__ import annotations

from torch.utils.data import DataLoader


def build_dataset(
    split: str,
    data_dir: str,
    dataset_name: str,
    history: int,
    horizon: int,
    train_stride: int,
):
    """One split's windows. val/test stride is forced to H by the dataset."""
    from mmtsfm.data.pv_record import PVRecordDataset

    return PVRecordDataset(
        split=split,
        dataset_name=dataset_name,
        data_path=data_dir,
        hist_steps=history,
        horizon=horizon,
        # Vision-free numerical control: no frame decode, no V-JEPA latents.
        emit_vision=False,
        # W4 cross-plant grouping is an MMTSFM mechanism with no iTransformer
        # counterpart; N=1 changes how windows are grouped into a sample, never
        # which windows exist.
        num_entities=1,
        # Future weather treated as known — PVRecordDataset's own default and
        # the assumption MMTSFM is trained under.
        future_cov="all",
        stride=train_stride if split == "train" else None,
    )


def build_loader(dataset, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=train,
        drop_last=train,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
