import os
import argparse
import numpy as np
from models.MultiModalTimer import MultiModalTimerConfig, MultiModalTimerModel, MultiModalTimerDataset
from transformers import Trainer, TrainingArguments
from safetensors.torch import load_file

class TextsCollator:
    def __call__(self, features):
        texts = [f["texts"] for f in features]
        
        from transformers import default_data_collator
        batch = default_data_collator([{k: v for k, v in f.items() if k != "texts"} for f in features])
        
        batch["texts"] = texts
        return batch

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # arguments for model config
    parser.add_argument('--forecasting_length', type=int)

    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--dataset_text', type=str)
    parser.add_argument('--checkpoint_path', type=str)

    args = parser.parse_args()
    
    if "CLIP" in args.checkpoint_path:
        vision_model_name = "CLIP"
    elif "BLIP" in args.checkpoint_path:
        vision_model_name = "BLIP"
    else:
        vision_model_name = None
    test_dataset = MultiModalTimerDataset(dataset_path=args.test_dataset_path, vision_model_name=vision_model_name, dataset_text=args.dataset_text, forecasting_length=args.forecasting_length)

    config = MultiModalTimerConfig.from_pretrained(os.path.join(args.checkpoint_path, 'config.json'))
    model = MultiModalTimerModel(config)
    state_dict = load_file(os.path.join(args.checkpoint_path, 'model.safetensors'))
    model.load_state_dict(state_dict, strict=False)

    training_args = TrainingArguments(
        output_dir="/home/ssh_adnlp/TSF/Vision_TSFM/ckpt/temp",
        disable_tqdm=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TextsCollator(),
    )

    output = trainer.predict(test_dataset)

    pred = output.predictions[:, :args.forecasting_length]
    true = output.label_ids

    mse = np.mean(np.square(pred-true))
    print(f"MSE: {mse:.4f}")