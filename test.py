import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PRETRAINED_MODEL1 = "cyberagent/open-calm-7b"
PRETRAINED_MODEL2 = "mosaicml/mpt-30b-chat"

model_name = PRETRAINED_MODEL1

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#↓プロンプト↓
prompt = "Q:「超弦理論」とは何ですか\nA:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )
    
output = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(output)