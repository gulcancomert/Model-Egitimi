# train_deit.py
import os, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import timm
from timm.data import create_transform, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2

from sklearn.metrics import accuracy_score, f1_score
from torch import amp

# =======================
# AYARLAR
# =======================
# Bu dosyanın yanındaki "data" klasörünü kullan (içinde train/val/test olmalı)
DATA_ROOT = str(Path(__file__).parent / "data")

MODEL_NAME = "deit_small_patch16_224"
IMG_SIZE = 224

BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOP_PATIENCE = 10

LR = 5e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1

MIXUP = 0.2
CUTMIX = 0.1
RAND_ERASE_P = 0.25

USE_EMA = True
EMA_DECAY = 0.9999

# Windows'ta sorun çıkmaması için 0 bırak (istersen __main__ guard ile artırabilirsin)
NUM_WORKERS = 0

OUTPUT_DIR = "./outputs_deit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = amp.GradScaler("cuda", enabled=(device == "cuda"))

# =======================
# Yol ve klasör kontrolleri
# =======================
p = Path(DATA_ROOT)
print(f"[Kontrol] DATA_ROOT = {p}")
missing = [d for d in ["train", "val", "test"] if not (p / d).exists()]
if missing:
    raise SystemExit(f"Şu klasör(ler) eksik: {missing}. DATA_ROOT içinde train/val/test olmalı!")

# =======================
# Dönüşümler (augment)
# =======================
train_tfms = create_transform(
    input_size=IMG_SIZE,
    is_training=True,
    auto_augment="rand-m9-n2",  # RandAugment
    re_prob=RAND_ERASE_P,       # Random Erasing
    interpolation="bicubic",
)
val_tfms = create_transform(
    input_size=IMG_SIZE,
    is_training=False,
    interpolation="bicubic",
)

# =======================
# Dataset & DataLoader
# =======================
train_ds = ImageFolder(p / "train", transform=train_tfms)
val_ds   = ImageFolder(p / "val",   transform=val_tfms)
test_ds  = ImageFolder(p / "test",  transform=val_tfms)

num_classes = len(train_ds.classes)
print(f"Sınıf sayısı: {num_classes}")
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Cihaz: {device} | Batch: {BATCH_SIZE} | LR: {LR}")

pin_mem = (device == "cuda")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=pin_mem)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_mem)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_mem)

# =======================
# Model, optimizer, scheduler, loss
# =======================
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes).to(device)

optimizer = create_optimizer_v2(model, opt="adamw", lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

warmup_t = max(5, int(EPOCHS * 0.1))
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=EPOCHS,        # toplam epoch
    lr_min=1e-6,             # en düşük lr
    warmup_lr_init=1e-6,     # warmup başlangıç lr
    warmup_t=warmup_t,       # warmup süresi (epoch)
    t_in_epochs=True
)

# Mixup/CutMix varsa eğitim için SoftTarget, yoksa LabelSmoothing kullan
mixup_fn = None
if MIXUP > 0.0 or CUTMIX > 0.0:
    mixup_fn = Mixup(
        mixup_alpha=MIXUP,
        cutmix_alpha=CUTMIX,
        label_smoothing=LABEL_SMOOTHING,
        num_classes=num_classes
    )
    train_criterion = SoftTargetCrossEntropy()  # eğitim
else:
    train_criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)  # eğitim

val_criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)  # val/test için sabit

ema = ModelEmaV2(model, decay=EMA_DECAY) if USE_EMA else None

# =======================
# Fonksiyonlar
# =======================
def train_one_epoch(epoch: int):
    model.train()
    running_loss, n = 0.0, 0

    t0 = time.time()
    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(images)
            loss = train_criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        bs = images.size(0)
        running_loss += loss.item() * bs
        n += bs

    avg_loss = running_loss / max(1, n)
    dt = time.time() - t0
    return avg_loss, dt


@torch.no_grad()
def evaluate(loader, use_ema=True):
    mdl = ema.module if (ema is not None and use_ema) else model
    mdl.eval()

    losses, n = 0.0, 0
    y_true, y_pred = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = mdl(images)
            loss = val_criterion(outputs, targets)

        bs = images.size(0)
        losses += loss.item() * bs
        n += bs

        preds = outputs.argmax(dim=1)
        # targets zaten int label; yine de güvenlik için:
        if targets.dim() > 1:
            targets_eval = targets.argmax(dim=1)
        else:
            targets_eval = targets

        y_true.extend(targets_eval.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = losses / max(1, n)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1

# =======================
# Eğitim döngüsü + Early Stopping
# =======================
best_val_loss = float("inf")
best_state = None
best_epoch = -1
no_improve = 0

for epoch in range(EPOCHS):
    train_loss, train_time = train_one_epoch(epoch)
    val_loss, val_acc, val_f1 = evaluate(val_loader, use_ema=True)

    # epoch sonunda LR scheduler
    scheduler.step(epoch + 1)

    improved = val_loss < best_val_loss - 1e-4
    if improved:
        best_val_loss = val_loss
        best_state = {
            "model": (ema.module if ema is not None else model).state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(best_state, os.path.join(OUTPUT_DIR, "best.pt"))
        best_epoch = epoch
        no_improve = 0
    else:
        no_improve += 1

    print(f"[{epoch+1:03d}/{EPOCHS}] "
          f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
          f"val_acc={val_acc:.4f}  val_f1(macro)={val_f1:.4f}  "
          f"time={train_time:.1f}s  "
          f"{'*BEST*' if improved else ''}")

    if no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping! {EARLY_STOP_PATIENCE} epoch iyileşme yok.")
        break

# =======================
# En iyi modeli yükle ve test et
# =======================
ckpt_path = os.path.join(OUTPUT_DIR, "best.pt")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    (model if ema is None else ema.module).load_state_dict(ckpt["model"])
    test_loss, test_acc, test_f1 = evaluate(test_loader, use_ema=True)
    print(f"\n[TEST] loss={test_loss:.4f} acc={test_acc:.4f} f1(macro)={test_f1:.4f}")
    print(f"En iyi epoch: {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
else:
    print("\nUyarı: best.pt bulunamadı, eğitim erken bitmiş olabilir.")
