import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import json
import gc
gc.collect()
torch.cuda.empty_cache()

USERNAME = "s2425823"
REMOTE_DIR = f"/home/{USERNAME}"

device = "cuda"

# путь к базовой модели "DAMO-NLP-SG/VideoLLaMA3-7B":
model_path = "/home/s2425823/.cache/huggingface/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B/snapshots/a498675483e2be8e98d092a2cb11a608c2caa8dd"

# путь к модели, обученной через LoRa:
tuned_model_path = "/home/s2425823/lora_videollama_finetuned_610_28_3epochs_speed10"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    local_files_only=True
)

processor = AutoProcessor.from_pretrained(model_path,
                                          trust_remote_code=True,
                                          local_files_only=True)

model = PeftModel.from_pretrained(model, tuned_model_path)

# Слияние LoRA-адаптеров с базовой моделью и выгрузка PEFT-компонентов (Объединение весов; опционально)
# чтобы модель стала "плоской" и могла использоваться как обычная HuggingFace-модель:
# model = model.merge_and_unload()

# Сохранение итоговой модели и токенизатора для последующего инференса
# save_dir = "path/to/exported_model"
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)


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
            {"type": "video", "video": {"video_path": REMOTE_DIR + "/" + video_path,
                                        "fps": 30, "max_frames": 300 }},
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
output_ids = model.generate(**inputs, max_new_tokens=4000)
response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

####################################################
# response = "TRAIN LORA MODEL:\n" + response
# del model
# del processor
# gc.collect()
# torch.cuda.empty_cache()

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     trust_remote_code=True,
#     device_map={"": device},
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     local_files_only=True
# )

# processor = AutoProcessor.from_pretrained(model_path,
#                                           trust_remote_code=True,
#                                           local_files_only=True)

# inputs = processor(
#     conversation=conversation,
#     add_system_prompt=True,
#     add_generation_prompt=True,
#     return_tensors="pt"
# )
# inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
# if "pixel_values" in inputs:
#     inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
# output_ids = model.generate(**inputs, max_new_tokens=4000)
# response2 = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

# response = response + "\n\nBASE MODEL:\n" + response2

####################################################


with open("finish.txt", "w", encoding="utf-8") as file:
    file.write(response)
