import argparse
import os
from models.MultiModalChronos import MultiModalChronosConfig, MultiModalChronosModel, MultiModalChronosDataset
from transformers import Trainer, TrainingArguments, set_seed
import torch
from safetensors.torch import save_file

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
    parser.add_argument('--vision_model_name', type=str)
    parser.add_argument('--vision_model_path', type=str)
    parser.add_argument('--text_model_name', type=str)
    parser.add_argument('--text_model_path', type=str)
    parser.add_argument('--chronos_path', type=str)
    parser.add_argument('--vision_model_prompt_len', type=int)
    parser.add_argument('--text_model_prompt_len', type=int)
    parser.add_argument('--chronos_prompt_len', type=int)


    # arguments for dataset
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_text', type=str)

    # arguments for training arg
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--train_epoch', type=int)

    args = parser.parse_args()

    random_seed = 42
    
    set_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    config = MultiModalChronosConfig(
        forecasting_length=args.forecasting_length,
        vision_model_name =args.vision_model_name,
        vision_model_path=args.vision_model_path,
        text_model_name =args.text_model_name,
        text_model_path=args.text_model_path,
        chronos_path=args.chronos_path,
        vision_model_prompt_len=args.vision_model_prompt_len,
        text_model_prompt_len=args.text_model_prompt_len,
        chronos_prompt_len=args.chronos_prompt_len,
        )
    model = MultiModalChronosModel(config)

    train_dataset = MultiModalChronosDataset(dataset_path=os.path.join(args.dataset_path, "train"), vision_model_name=args.vision_model_name, dataset_text=args.dataset_text, forecasting_length=args.forecasting_length)
    val_dataset = MultiModalChronosDataset(os.path.join(args.dataset_path, "val"), vision_model_name=args.vision_model_name, dataset_text=args.dataset_text, forecasting_length=args.forecasting_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=args.train_epoch,
        logging_strategy="epoch",
        logging_first_step=True,
        disable_tqdm=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=TextsCollator(),
    )

    trainer.train()

    state_dict = model.state_dict()

    essential_state_dict = {
        k: v for k, v in state_dict.items() if 'prompts' in k or 'interaction' in k
    }

    model.config.save_pretrained(args.output_dir)
    save_file(essential_state_dict, os.path.join(args.output_dir, "model.safetensors"))