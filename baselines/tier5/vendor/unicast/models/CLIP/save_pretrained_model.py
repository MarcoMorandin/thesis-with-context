from transformers import AutoModel
from safetensors.torch import save_file

model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")

model.vision_model.config.save_pretrained("./")

state_dict = model.vision_model.state_dict()
save_file(state_dict, "./model.safetensors")