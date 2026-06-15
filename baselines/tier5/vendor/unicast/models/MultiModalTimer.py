import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PretrainedConfig
from safetensors.torch import load_file

# CLIP
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from CLIP.modeling_clipPT import CLIPVisionTransformer
from transformers import CLIPImageProcessor

# BLIP
from transformers.models.blip.configuration_blip import BlipVisionConfig
from BLIP.modeling_blipPT import BlipVisionModel
from transformers import BlipImageProcessor

from transformers import AutoTokenizer

# Qwen
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from Qwen.modeling_qwen2 import Qwen2Model

# LLaMA
from transformers.models.llama.configuration_llama import LlamaConfig
from LLaMA.modeling_llama import LlamaModel

# Timer
from Timer.configuration_timer import TimerConfig
from Timer.modeling_timer import TimerForPrediction

class MultiModalTimerConfig(PretrainedConfig):
    def __init__(
        self,
        forecasting_length = None,
        vision_model_name = None,
        vision_model_path = None,
        text_model_name = None,
        text_model_path = None,
        timer_path = None,
        vision_model_prompt_len = None,
        text_model_prompt_len = None,
        timer_prompt_len = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.forecasting_length = forecasting_length
        self.vision_model_name = vision_model_name
        self.vision_model_path = vision_model_path
        self.text_model_name = text_model_name
        self.text_model_path = text_model_path
        self.timer_path = timer_path
        
        self.vision_model_prompt_len = vision_model_prompt_len if vision_model_prompt_len is not None else 10
        self.text_model_prompt_len = text_model_prompt_len if text_model_prompt_len is not None else 4
        self.timer_prompt_len = timer_prompt_len if timer_prompt_len is not None else 4

class MultiModalTimerModel(PreTrainedModel):
    
    config_class = MultiModalTimerConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Vision Model
        if config.vision_model_name is None:
            pass
        elif config.vision_model_name == 'CLIP':
            vision_model_config = CLIPVisionConfig.from_pretrained(os.path.join(config.vision_model_path, 'config.json'))
            self.vision_model = CLIPVisionTransformer(vision_model_config, config.vision_model_prompt_len)
            state_dict = load_file(os.path.join(config.vision_model_path, 'model.safetensors'))
            self.vision_model.load_state_dict(state_dict, strict=False)
            for name, param in self.vision_model.named_parameters(): # Freeze layers other than prompts
                if "encoder.prompts" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif config.vision_model_name == 'BLIP':
            vision_model_config = BlipVisionConfig.from_pretrained(os.path.join(config.vision_model_path, 'config.json'))
            self.vision_model = BlipVisionModel(vision_model_config, config.vision_model_prompt_len)
            state_dict = load_file(os.path.join(config.vision_model_path, 'model.safetensors'))
            self.vision_model.load_state_dict(state_dict, strict=False)
            for name, param in self.vision_model.named_parameters(): # Freeze layers other than prompts
                if "encoder.prompts" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            pass
        
        # Text Model
        if config.text_model_name is None:
            pass
        elif config.text_model_name == 'Qwen':
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
            text_model_config = Qwen2Config.from_pretrained(os.path.join(config.text_model_path, 'config.json'))
            self.text_model = Qwen2Model(text_model_config, config.text_model_prompt_len)
            state_dict = load_file(os.path.join(config.text_model_path, 'model.safetensors'))
            self.text_model.load_state_dict(state_dict, strict=False)
            for name, param in self.text_model.named_parameters(): # Freeze layers other than prompts
                if "prompts" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif config.text_model_name == 'LLaMA':
            self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
            text_model_config = LlamaConfig.from_pretrained(os.path.join(config.text_model_path, 'config.json'))
            self.text_model = LlamaModel(text_model_config, config.text_model_prompt_len)
            state_dict = {}
            for i in range(1, 4):
                path = os.path.join(config.text_model_path, f"model-0000{i}-of-00003.safetensors")
                partial_state = load_file(path)
                state_dict.update(partial_state)
            self.text_model.load_state_dict(state_dict, strict=False)
            for name, param in self.text_model.named_parameters(): # Freeze layers other than prompts
                if "prompts" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            pass
            
        # Timer
        timer_config = TimerConfig.from_pretrained(os.path.join(config.timer_path, 'config.json'))
        self.timer = TimerForPrediction(timer_config, config.timer_prompt_len)
        state_dict = load_file(os.path.join(config.timer_path, 'model.safetensors'))
        self.timer.load_state_dict(state_dict, strict=False)
        for name, param in self.timer.named_parameters(): # Freeze layers other than prompts
            if "model.prompts" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Vision Interaction Layer
        if config.vision_model_name is None:
            pass
        else:
            self.vision_interaction_layer = nn.Linear(self.vision_model.config.hidden_size, timer_config.hidden_size)

        # Text Interaction Layer
        if config.text_model_name is None:
            pass
        else:
            self.text_interaction_layer = nn.Linear(self.text_model.config.hidden_size, timer_config.hidden_size)
    
    def forward(self, input_ids = None, images = None, texts = None, labels = None):
        if self.config.vision_model_name is None and images is None:
            vision_embedding = None
        else:
            vision_embedding = self.vision_model(images)
            vision_embedding = vision_embedding.pooler_output
            vision_embedding = self.vision_interaction_layer(vision_embedding)

        if self.config.text_model_name is None and all(x is None for x in texts):
            text_embedding = None
        else:
            tokenized_texts = self.tokenizer(texts, return_tensors="pt").to("cuda")
            text_embedding = self.text_model(**tokenized_texts)
            text_embedding = text_embedding.last_hidden_state[:, 0 , :]
            text_embedding = self.text_interaction_layer(text_embedding)

        out = self.timer(input_ids=input_ids, vision_embedding=vision_embedding, text_embedding=text_embedding)
        out = out["logits"]

        if labels is not None:
            if self.config.forecasting_length == out.shape[-1]:
                loss = torch.mean(torch.square(out-labels)) # MSE
            else: # pretrained Timer has 96 forecasting length. This is in case of shorter forecasting length. Forecasting length larger than 96 will occur an error.
                loss = torch.mean(torch.square(out[:, :self.config.forecasting_length]-labels))
        else:
            loss = None

        return {
            "loss": loss,
            "logits": out
        }

class MultiModalTimerDataset(Dataset):
    def __init__(self, dataset_path, vision_model_name = None, dataset_text = None, forecasting_length: int = 96):
        
        self.dataset_path = dataset_path
        self.vision_model_name = vision_model_name
        self.dataset_text = dataset_text

        if vision_model_name is None:
            pass
        elif vision_model_name == 'CLIP':
            self.processor = CLIPImageProcessor()
        elif vision_model_name == 'BLIP':
            self.processor = BlipImageProcessor()
        else:
            pass

        self.inputs = torch.load(os.path.join(dataset_path, "inputs.pt"))
        if forecasting_length:
            self.targets = torch.load(os.path.join(dataset_path, f"targets_{forecasting_length}.pt"))
        else:
            self.targets = torch.load(os.path.join(dataset_path, "targets.pt"))
        self.keys = list(self.targets.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_name = self.keys[idx]
        
        if self.vision_model_name is None:
            images = None
        else:
            img_path = os.path.join(self.dataset_path, 'img', img_name)
            images = Image.open(img_path).convert("RGB")
            images = self.processor.preprocess(images)['pixel_values'][0]

        input_tensor = self.inputs[img_name].float().squeeze()
        target_tensor = self.targets[img_name].float().squeeze()

        return {
            "input_ids": input_tensor,
            "images": images,
            "texts": self.dataset_text,
            "labels": target_tensor
        }