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

# Chronos
from transformers.models.t5.modeling_t5 import T5Config
from Chronos.chronos_bolt import ChronosBoltModelForForecasting

class MultiModalChronosConfig(PretrainedConfig):
    def __init__(
        self,
        forecasting_length = None,
        vision_model_name = None,
        vision_model_path = None,
        text_model_name = None,
        text_model_path = None,
        chronos_path = None,
        vision_model_prompt_len = None,
        text_model_prompt_len = None,
        chronos_prompt_len = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.forecasting_length = forecasting_length
        self.vision_model_name = vision_model_name
        self.vision_model_path = vision_model_path
        self.text_model_name = text_model_name
        self.text_model_path = text_model_path
        self.chronos_path = chronos_path
        
        self.vision_model_prompt_len = vision_model_prompt_len if vision_model_prompt_len is not None else 10
        self.text_model_prompt_len = text_model_prompt_len if text_model_prompt_len is not None else 4
        self.chronos_prompt_len = chronos_prompt_len if chronos_prompt_len is not None else 4

class MultiModalChronosModel(PreTrainedModel):
    
    config_class = MultiModalChronosConfig

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

        # Chronos
        chronos_config = T5Config.from_pretrained(os.path.join(config.chronos_path, 'config.json'))
        self.chronos = ChronosBoltModelForForecasting(chronos_config, config.chronos_prompt_len)
        state_dict = load_file(os.path.join(config.chronos_path, 'model.safetensors'))
        self.chronos.load_state_dict(state_dict, strict=False)
        for name, param in self.chronos.named_parameters(): # Freeze layers other than prompts
            if "encoder.prompts" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Vision Interaction Layer
        if config.vision_model_name is None:
            pass
        else:
            self.vision_interaction_layer = nn.Linear(self.vision_model.config.hidden_size, chronos_config.d_model)
        
        # Text Interaction Layer
        if config.text_model_name is None:
            pass
        else:
            self.text_interaction_layer = nn.Linear(self.text_model.config.hidden_size, chronos_config.d_model)
    
    def predict(  # type: ignore[override]
        self,
        context: torch.Tensor,
        vision_embedding: torch.Tensor = None,
        text_embedding: torch.Tensor = None,
        prediction_length: int = None,
    ) -> torch.Tensor:
        """
        Get forecasts for the given time series.

        Refer to the base method (``BaseChronosPipeline.predict``)
        for details on shared parameters.

        Returns
        -------
        torch.Tensor
            Forecasts of shape (batch_size, num_quantiles, prediction_length)
            where num_quantiles is the number of quantiles the model has been
            trained to output. For official Chronos-Bolt models, the value of
            num_quantiles is 9 for [0.1, 0.2, ..., 0.9]-quantiles.
        """
        context_tensor = context

        model_context_length = self.chronos.config.chronos_config["context_length"]
        model_prediction_length = self.chronos.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance collapsing
        # every 64 steps.
        context_tensor = context_tensor.to(
            device=self.chronos.device,
            dtype=torch.float32,
        )
        while remaining > 0:
            prediction = self.chronos(
                context=context_tensor, vision_embedding=vision_embedding, text_embedding=text_embedding,
            ).quantile_preds.to(context_tensor)
            
            central_idx = torch.abs(torch.tensor(self.chronos.config.chronos_config["quantiles"]) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            predictions.append(central_prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32
        )
    
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

        out = self.predict(context=input_ids, vision_embedding=vision_embedding, text_embedding=text_embedding, prediction_length=self.config.forecasting_length)

        if labels is not None:
            loss = torch.mean(torch.square(out-labels)) # MSE
        else:
            loss = None

        return {
            "loss": loss,
            "logits": out
        }

class MultiModalChronosDataset(Dataset):
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