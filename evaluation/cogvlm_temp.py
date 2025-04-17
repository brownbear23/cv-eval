import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

device_used = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-grounding-generalist-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device_used).eval()

query = 'Can you provide a description of the image and include the coordinates [[x0,y0,x1,y1]] for each mentioned object?'
image = Image.open("img.jpg").convert("RGB")
inputs = model.build_conversation_input_ids(tokenizer, query=query, images=[image])
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device_used),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device_used),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))
