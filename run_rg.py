from transformers import RecurrentGemmaConfig, AutoTokenizer, RecurrentGemmaForCausalLM
import torch
import os 

HF_TOKEN = os.getenv("HF_TOKEN")

print('loading rg config')
config = RecurrentGemmaConfig.from_pretrained("google/recurrentgemma-2b-it", token=HF_TOKEN)
print('loading rg tokenizer')
tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b-it",  token=HF_TOKEN)

print('loading rg model')
pretrained_model = RecurrentGemmaForCausalLM.from_pretrained("google/recurrentgemma-2b-it", token=HF_TOKEN, output_attentions=True)
inputs = tokenizer.encode("The cat sat on the mat", return_tensors='pt')
outputs = pretrained_model(inputs)
print(outputs)
