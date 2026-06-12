import sys
sys.path.insert(0, './src')

from mmtsfm.data.datamodule import MMTSFMDataModule

def test_dataloader(dataset_name: str = "synthetic", split: str = "fit"):
    dm = MMTSFMDataModule(
        data_dir="./data",
        dataset_name=dataset_name,
        num_entities=10,
        hist_steps=24,
        horizon=12,
        target_dim=1,
        covariate_dim=5,
        video_frames=8,
        img_size=64,
        batch_size=16,
    )
    dm.setup(split)

    if split == "test":
        loader = dm.test_dataloader()
    else:
        loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    print("Dataset:", dataset_name)
    print("Batch keys:", list(batch.keys()))
    print("Y shape:", batch['Y'].shape)
    print("X_cov shape:", batch['X_cov'].shape)
    print("V shape:", batch['V'].shape)
    print("timestamps shape:", batch['timestamps'].shape)
    print("entity_ids shape:", batch['entity_ids'].shape)
    print("timestamps_v shape:", batch['timestamps_v'].shape)
    print("mask_target shape:", batch['mask_target'].shape)
    print("mask_visual shape:", batch['mask_visual'].shape)
    print("mask_modality_dropout shape:", batch['mask_modality_dropout'].shape)
    print("adj_matrix shape:", batch['adj_matrix'].shape)
    
    print("Successfully sampled from the Multimodal Dataloader!")

if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    split = sys.argv[2] if len(sys.argv) > 2 else "fit"
    test_dataloader(dataset_name=dataset_name, split=split)
