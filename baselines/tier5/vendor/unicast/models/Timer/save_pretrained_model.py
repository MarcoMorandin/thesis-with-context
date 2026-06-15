from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)

model.save_pretrained("./")