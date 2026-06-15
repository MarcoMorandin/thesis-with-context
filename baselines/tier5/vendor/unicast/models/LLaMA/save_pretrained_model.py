import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

model.model.save_pretrained("./")