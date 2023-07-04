import argparse
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

PRETRAINED_MODEL1 = "cyberagent/open-calm-7b"
PRETRAINED_MODEL2 = "mosaicml/mpt-30b-chat"
PRETRAINED_MODEL3 = "tiiuae/falcon-40b-instruct"

def relativeword(args):
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    word = args.search_word
    template = "Explain {}."
    prompt = template.format(word)
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        tokens = model.generate(
        **inputs,
        max_new_tokens=90,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        length_penalty=1.00,
        pad_token_id=tokenizer.pad_token_id,
    )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(output)

if __name__ == '__main__':
    model_path = "tiiuae/falcon-40b-instruct"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="tiiuae/falcon-40b-instruct", help="model path")
    parser.add_argument("--search_word", type=str, default="Halcyon", help="related words you want to explore")
    args = parser.parse_args()

    relativeword(args)