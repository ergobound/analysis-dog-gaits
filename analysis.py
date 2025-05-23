import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import json
import gc
gc.collect()
torch.cuda.empty_cache()

USERNAME = "s2425823"
REMOTE_DIR = f"/home/{USERNAME}/"

device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    
video_path = data.get("video_path")
prompt = data.get("prompt")
text = data.get("text")

conversation = [
    {"role": "system", "content": prompt},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": REMOTE_DIR + video_path,
                                        "fps": 1, "max_frames": 180}},
            {"type": "text", "text": text},
        ]
    },
]

inputs = processor(
    conversation=conversation,
    add_system_prompt=True,
    add_generation_prompt=True,
    return_tensors="pt"
)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
output_ids = model.generate(**inputs, max_new_tokens=8000)
response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

with open("finish.txt", "w", encoding="utf-8") as file:
    file.write(response)
