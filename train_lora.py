from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    AutoProcessor,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import os
import json
import cv2
import numpy as np
import torch
import gc
gc.collect()
torch.cuda.empty_cache()

# Пути к модели:
model_path = "/home/s2425823/.cache/huggingface/hub/models--DAMO-NLP-SG--VideoLLaMA3-7B/snapshots/a498675483e2be8e98d092a2cb11a608c2caa8dd"
# Путь к обученой через lora модели
path_finetuned_model = "/home/s2425823/lora_videollama_finetuned_610v3_DubiDub" 
# Путь к файлам настройки тренировки:
out_dir = "/home/s2425823/dog_gait_lora_610v3_DubiDub"

json_path = "/home/s2425823/dataset610/train.json" # путь к train датасету
# val_path = "/home/s2425823/dataset610/val.json" # путь к validate датасету

USERNAME = "s2425823"
REMOTE_DIR = f"/home/{USERNAME}"
device = "cuda"

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
    def __init__(self, json_path, tokenizer, video_processor_fn, max_length=256):
        with open(json_path, "r", encoding="utf8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.video_processor = video_processor_fn
        self.max_length = max_length

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
            return_tensors="pt"
        )

        # Токенизируем только промпт, чтобы определить его длину
        tokenized_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
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

def main():
    # Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                            trust_remote_code=True,
                                            local_files_only=True)
    # trust_remote_code=True, # если True, используется базовый класс, например VideoLLaMAForCausalLM, внутри которого переопределён метод .generate(), и в нём по умолчанию задан temperature=0.7 и скорее всего другие параметры. Это поведение задаётся в коде самой модели, которую стоит, например DAMO-NLP-SG/VideoLLaMA3-7B
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                            device_map="auto", 
                                            trust_remote_code=True,
                                            local_files_only=True)
    # Оптимизация памяти или вроде того:
    model.gradient_checkpointing_enable()
    # model.half()

    # Настройка LoRa через peft.
    lora_config = LoraConfig(
        # Ранг низкоранговой матрицы — чем больше, тем больше параметров у LoRA, тем мощнее адаптация. Обычно 4–16:
        r=4, # 8 или 16
        # Масштаб - множитель для итоговой матрицы, влияет на "громкость" влияния LoRA. Чатсто ставят r * 2 :
        lora_alpha=8, # 32
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
    )

    # Задаём параметры обучения
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3, # Количество эпох (слишком много эпох переучит модель)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-10, # Начальная скорость обучения
        fp16=False, # fp16=True - используем половинную точность
        logging_steps=10, # Каждые N шагов выводить логи (loss, lr и пр.)
        save_steps=100, # Каждые N шагов сохранять чекпоинт модели
        save_total_limit=2, # 2; Хранить не более N чекпоинтов, старые удаляются
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Обязательно для видео (Использует контрольные точки градиента -> экономит VRAM, особенно при видео)
        save_strategy="epoch", # Сохраняет модель в конце каждой эпохи
        logging_dir="./logs",  # Куда писать Tensor логи (tensorboard -- logdir=...)
    )

    # Создаём Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset, # тренировочный датасет
        data_collator=collate_fn
    )

    # Запускаем обучение
    trainer.train()
    # Сохраняем дообученную модель
    model.save_pretrained(path_finetuned_model)

main()