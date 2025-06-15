import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback
import gc
gc.collect()
torch.cuda.empty_cache()

# Функция для обработки видео.
def video_processor(video_path, num_frames=180, size=(610, 610)): # 224, 224
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Не удалось определить количество кадров в видео: {video_path}")
    # Выбираем равномерно распределённые кадры
    frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    current_frame = 0
    ret = True

    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in frame_ids:
            # Изменяем размер каждого кадра на size, который по умолчанию (244,244) или (610,610)
            resized_frame = cv2.resize(frame, size)
            # Преобразуем BGR (OpenCV) в RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        current_frame += 1

    cap.release()
    if not frames:
        raise ValueError(f"Не удалось извлечь кадры из видео: {video_path}")
    frames = np.array(frames).astype(np.float32) / 255.0  # нормализация
    # Меняем размерность: (num_frames, 3, height, width)
    frames = torch.tensor(frames).permute(0, 3, 1, 2)
    return frames

# Кастомный датасет для VideoLLaMA.
class VideoLLaMADataset(Dataset):
    def __init__(self, json_path, tokenizer, video_processor_fn, max_length=256, min_length=256):
        with open(json_path, "r", encoding="utf8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.video_processor = video_processor_fn
        self.max_length = max_length
        self.min_length = min_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = ""
        response = ""
        video_frames = None

        # Формируем единую последовательность: сначала текст пользователя с placeholder для видео, затем ответ ассистента.
        for turn in sample["conversations"]:
            role = turn.get("role", "")
            if role == "user":
                for content_item in turn.get("content", []):
                    if content_item.get("type") == "video":
                        video_path = content_item["video"]["video_path"]
                        # Извлекаем кадры видео
                        video_frames = self.video_processor(video_path)
                        # Вместо видео вставляем специальный токен
                        prompt += " [VIDEO] "
                    elif content_item.get("type") == "text":
                        prompt += content_item.get("text", "") + " "
            elif role == "assistant":
                response += turn.get("content", "") + " "

        # Объединяем промпт(видео+запрос) и ответ
        full_text = prompt + response

        # Токенизируем полную последовательность
        tokenized_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            # min_length=self.min_length,
            return_tensors="pt"
        )

        # Токенизируем только промпт, чтобы определить его длину
        tokenized_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            # min_length=self.min_length,
            return_tensors="pt"
        )
        prompt_length = tokenized_prompt["input_ids"].shape[1]

        # Копируем токенизированные последовательности в метки
        labels = tokenized_full["input_ids"].clone()
        # Маскируем токены, относящиеся к промпту; потерь не считаем по ним
        labels[:, :prompt_length] = -100

        item = {
            "input_ids": tokenized_full["input_ids"].squeeze(0),    # squeeze метод удаляет размер батча, равный 1
            "attention_mask": tokenized_full["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
        if video_frames is not None:
            item["video_frames"] = video_frames  # дополнительно передаём видео, если оно есть
            item["video_path"] = video_path # добавляем еще и путь к видео, чтобы в будущем отслеживать
        return item

# collate_fn — это функция-склейщик. Она выполняет правильное объединение разного размера примеров в единый батч.
# Батч (batch) — это просто группа примеров, которые модель обрабатывает одновременно на одном шаге обучения. Это нужно, чтобы ускорить обучение и сделать его более стабильным.

def collate_fn(batch):
    # Объединяем текстовые данные
    batch_input = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=0
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=-100
        )
    }
    # Если видео присутствует, объединяем и их
    if "video_frames" in batch[0]:
        batch_input["video_frames"] = torch.stack([b["video_frames"] for b in batch])
    return batch_input

# Callback класс для постоянного мониторинга памяти на gpu
class GpuMonitorCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n[GPU usage] Выделено: {allocated:.2f} GB | Зарезервировано: {reserved:.2f} GB")

# greedy_decode для класса валидации
def greedy_decode(model, tokenizer, input_ids, attention_mask, video_frames=None, max_new_tokens=200):
    model.eval()
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                video_frames=video_frames  # <- если поддерживается моделью
            )
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # только последний токен
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # (batch_size, 1)

            # Если модель предсказала [EOS] — можно завершать
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Добавим следующий токен к уже сгенерированным
            generated = torch.cat([generated, next_token_id], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token_id)], dim=-1
            )

    return generated


# Мониторинг валидации
class EvalSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, num_samples=2):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.num_samples = num_samples

    def on_epoch_end(self, args, state, control, **kwargs):
        print("\n🌟 [Инференс по валидации] Примеры после эпохи", int(state.epoch))
        model = kwargs["model"]
        model.eval()

        for i in range(min(self.num_samples, len(self.val_dataset))):
            # print('VAR', self.val_dataset[i])
            sample = self.val_dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)
            video_frames = sample.get("video_frames", None)
            # if video_frames is not None:
            video_frames = video_frames.unsqueeze(0).to(model.device)

            # input_ids = input_ids[:, -1024:]
            # attention_mask = attention_mask[:, -1024:]

            # соединяем текст и видео в одно
            output_ids = greedy_decode(
                model=model,
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                video_frames=video_frames,
                max_new_tokens=2000
            )

            # Генерация ответа
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # video_frames=video_frames, # не работает
                    max_new_tokens=2000,
                    temperature=None, # игнорируется при do_sample=False
                    top_k=None, # игнорируется при do_sample=False
                    top_p=None, # игнорируется при do_sample=False
                    do_sample=False, # False - даёт более стабильную картинку при отслеживании качества модели

                    # generate() может упасть из-за отсутствие следующих двух параметров:
                    # pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    # eos_token_id=self.tokenizer.eos_token_id
                )

            # Декодируем и выводим
            prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            video_name = sample.get("video_path", "N/A").split("/")[-1]

            print(f"\n🎬 Видео: {video_name}")
            print(f"\n🧾 Запрос:\n{prompt_text.strip()}")
            print(f"\n🤖 Ответ модели:\n{output_text.strip()}")
            print("Output token IDs:", output_ids[0][:5])
            token = self.tokenizer.convert_ids_to_tokens(151645)
            print("Токен:", token)

path_finetuned_model = "/home/s2425823/lora_videollama_finetuned_610_17"
out_dir = "/home/s2425823/dog_gait_lora_610_17"
val_path = "/home/s2425823/dataset610/val.json"

def main():
    # Пути к данным и модели
    json_path = "/home/s2425823/dataset610/train.json"
    model_name_or_path = "/home/s2425823/.cache/huggingface/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B/snapshots/a498675483e2be8e98d092a2cb11a608c2caa8dd"

    # Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                            trust_remote_code=True,
                                            local_files_only=True)
    # trust_remote_code=True, # если True, используется базовый класс, например VideoLLaMAForCausalLM, внутри которого переопределён метод .generate(), и в нём по умолчанию задан temperature=0.7 и скорее всего другие параметры. Это поведение задаётся в коде самой модели, которую стоит, например DAMO-NLP-SG/VideoLLaMA3-7B
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                            device_map="auto", 
                                            trust_remote_code=True,
                                            local_files_only=True)

    model.gradient_checkpointing_enable()
    # model.half()

    # Настройка LoRa через peft.
    # Укажите target_modules, соответствующие архитектуре вашей модели
    lora_config = LoraConfig(
        # Ранг низкоранговой матрицы — чем больше, тем больше параметров у LoRA, тем мощнее адаптация. Обычно 4–16:
        r=8, # 8 или 16
        # Масштаб - множитель для итоговой матрицы, влияет на "громкость" влияния LoRA. Чатсто ставят r * 2 :
        lora_alpha=16, # 32
        # Dropout между A и B матрицами (обычно 0.05 или 0.1):
        lora_dropout=0.1, # 0.1 или 0.05
        # Обучаются ли смещения в слоях:
        # bias="lora_only",
        # указываем, для какой архитектуры и задачи будет применяться адаптивное обучение (LoRA):
        task_type="CAUSAL_LM", # модель с причинным обучением (Causal Language Modeling)
        # target_modules=["q_proj", "v_proj"]
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()  # опционально для проверки параметров

    # Создаём датасет
    dataset = VideoLLaMADataset(
        json_path=json_path,
        tokenizer=tokenizer,
        video_processor_fn=video_processor,
        max_length=2000,
        # min_length=1200
    )

    val_dataset = VideoLLaMADataset(
        json_path=val_path,
        tokenizer=tokenizer,
        video_processor_fn=video_processor,
        max_length=2000
    )

    # Задаём параметры обучения
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3, # 3, 30; Количество эпох (слишком много эпох переучит модель)
        per_device_train_batch_size=2, # 1 2 
        gradient_accumulation_steps=4, # 8 4
        learning_rate=1e-4, # 5e-5 или 1e-4 # Начальная скорость обучения
        fp16=False, # fp16=True - используем половинную точность
        logging_steps=10, # Каждые N шагов выводить логи (loss, lr и пр.)
        save_steps=100, # Каждые N шагов сохранять чекпоинт модели
        save_total_limit=2, # 2; Хранить не более N чекпоинтов, старые удаляются
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Обязательно для видео (Использует контрольные точки градиента -> экономит VRAM, особенно при видео)
        # optim="adafactor",  # Экномоия памяти; "adamw_torch" или "adafactor", первый более стабилен 
        eval_strategy="epoch", # Запускать evalution в конце каждой эпохи
        save_strategy="epoch", # Сохраняет модель в конце каждой эпохи

        # eval_steps=500, # Валидация каждые N шагов
        logging_dir="./logs",  # Куда писать Tensor логи (tensorboard -- logdir=...)

        # lr_scheduler_type="cosine", # Стабильность
        # warmup_steps=50 # Дает чуть мягче старт
    )
    
    gpu_callback = GpuMonitorCallback()
    # eval_callback = EvalSampleCallback(tokenizer=tokenizer, val_dataset=val_dataset, num_samples=4)

    # Создаём Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset, # тренировочный датасет
        eval_dataset=val_dataset, # валидационный датасет
        data_collator=collate_fn,
        callbacks=[gpu_callback] # eval_callback
    )

    # Запускаем обучение
    trainer.train()
    # Сохраняем дообученную модель
    model.save_pretrained(path_finetuned_model)
main()


######################################################

from transformers import AutoProcessor
import json
import gc
gc.collect()
torch.cuda.empty_cache()

USERNAME = "s2425823"
REMOTE_DIR = f"/home/{USERNAME}"

device = "cuda"
model_path = "/home/s2425823/.cache/huggingface/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B/snapshots/a498675483e2be8e98d092a2cb11a608c2caa8dd"
# model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"

base_model = AutoModelForCausalLM.from_pretrained(
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

model2 = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    local_files_only=True
)

processor2 = AutoProcessor.from_pretrained(model_path,
                                          trust_remote_code=True,
                                          local_files_only=True)

from peft import PeftModel
model = PeftModel.from_pretrained(base_model, path_finetuned_model)
# Объединение весов (опционально):
model = model.merge_and_unload()

with open("/home/s2425823/dataset610/val_test.json", "r", encoding="utf-8") as file:
    data = json.load(file)


for part in data:
    conversation = part.get("conversations")
    content = conversation[0].get("content")
    video_link = content[0]
    user_text = content[1]

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
    print('\n\n')
    print(video_link)
    print("USER:\n", user_text)
    print('\nTRAIN MODEL.\n')
    print("ASSISTENT:\n", response)

    inputs = processor2(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model2.generate(**inputs, max_new_tokens=4000)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print('\nBASE MODEL.\n')
    print("ASSISTENT:\n", response)