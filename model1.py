from transformers import AutoImageProcessor, AutoModelForImageClassification
model_id = "google/vit-base-patch16-224"  # 또는 위 소형/DeiT로 교체
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

import numpy as np, librosa
from PIL import Image

SR, N_MELS, N_FFT, HOP = 16000, 128, 400, 160  # 25ms/10ms 정도

def wav_to_melspec_image(path, target_sec=4):
    y, sr = librosa.load(path, sr=SR)
    T = SR*target_sec
    if len(y) < T: y = np.pad(y, (0, T-len(y)))
    elif len(y) > T:
        s = (len(y)-T)//2
        y = y[s:s+T]

    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)

    # 0~255 정규화 → 3채널
    S_min, S_max = S_db.min(), S_db.max()
    arr = np.zeros_like(S_db, dtype=np.uint8) if S_max-S_min < 1e-6 \
        else ((S_db - S_min)/(S_max - S_min) * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB").resize((224,224))
    return img

import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

model_id = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

device = 0 if torch.cuda.is_available() else (-1 if not torch.backends.mps.is_available() else 0)
img = wav_to_melspec_image("sample.wav")

clf = pipeline("image-classification", model=model, feature_extractor=processor, device=device)
print(clf(img, top_k=5))

import os, torch, pandas as pd
from datasets import Dataset, load_dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from PIL import Image

# 1) 파일 목록을 (path, label) 형태로 수집
def scan_folder(root):
    paths, labels = [], []
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            if fn.lower().endswith(".wav"):
                paths.append(os.path.join(d, fn))
                labels.append(label)
    return paths, labels

train_paths, train_labels = scan_folder("data/train")
val_paths,   val_labels   = scan_folder("data/val")

le = LabelEncoder().fit(train_labels + val_labels)
id2label = {i:c for i,c in enumerate(le.classes_)}
label2id = {c:i for i,c in enumerate(le.classes_)}

train_ds = Dataset.from_dict({"path": train_paths, "label": [label2id[x] for x in train_labels]})
val_ds   = Dataset.from_dict({"path": val_paths,   "label": [label2id[x] for x in val_labels]})

# 2) ViT processor
model_id = "google/vit-base-patch16-224"  # patch16 = stride16 = 비중첩
processor = AutoImageProcessor.from_pretrained(model_id)

def preprocess(batch):
    img = wav_to_melspec_image(batch["path"])  # 위에서 정의한 함수
    enc = processor(images=img, return_tensors="pt")
    batch["pixel_values"] = enc["pixel_values"][0]
    batch["labels"] = int(batch["label"])
    return batch

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(preprocess,   remove_columns=val_ds.column_names)
train_ds.set_format(type="torch", columns=["pixel_values","labels"])
val_ds.set_format(type="torch",   columns=["pixel_values","labels"])

# 3) 모델 & 학습
model = AutoModelForImageClassification.from_pretrained(
    model_id,
    num_labels=len(le.classes_),
    id2label={i:str(s) for i,s in id2label.items()},
    label2id=label2id
)

args = TrainingArguments(
    output_dir="vit_ser_ckpt",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.05,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none"
)

import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return metric.compute(predictions=preds, references=p.label_ids)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
trainer.train()
trainer.evaluate()
trainer.save_model("vit_ser_model")
processor.save_pretrained("vit_ser_model")

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("vit_ser_model")
model = AutoModelForImageClassification.from_pretrained("vit_ser_model")
model.eval()
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

img = wav_to_melspec_image("test.wav")
enc = processor(images=img, return_tensors="pt")
with torch.no_grad():
    logits = model(**{k:v.to(device) for k,v in enc.items()}).logits
pred = logits.softmax(-1).argmax(-1).item()
print("pred:", model.config.id2label[str(pred)] if isinstance(model.config.id2label, dict) else model.config.id2label[pred])