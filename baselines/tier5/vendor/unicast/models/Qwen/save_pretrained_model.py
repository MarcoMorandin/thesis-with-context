import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)

model.model.save_pretrained("./")