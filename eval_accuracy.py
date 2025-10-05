
import argparse, json, os, torch, timm
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.data import create_transform
import numpy as np
from sklearn.metrics import f1_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/test", help="test klasörü (class alt klasörleri ile)")
    ap.add_argument("--ckpt", default="outputs_deit/best.pt", help="eğitimden çıkan checkpoint")
    ap.add_argument("--classes-json", default="app/data/classes.json")
    ap.add_argument("--model-name", default="deit_small_patch16_224")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--tta", action="store_true", help="Basit TTA (flip) kullan")
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device in ["auto","cuda"] and torch.cuda.is_available()) else "cpu")

 
    with open(args.classes_json, "r", encoding="utf-8") as f:
        cls_map = json.load(f)
    classes = [cls_map[str(i)] for i in range(len(cls_map))]

 
    tfm = create_transform(input_size=args.img_size, is_training=False, interpolation="bicubic")
    ds = ImageFolder(args.data_root, transform=tfm)
    assert ds.classes == classes, "ImageFolder sınıf sırası classes.json ile aynı olmalı!"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(dev.type=="cuda"))

   
    model = timm.create_model(args.model_name, pretrained=False, num_classes=len(classes))
    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()

    y_true, y_pred = [], []
    top5_correct, n = 0, 0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)

            if args.tta:
                logits1 = model(x)
                logits2 = model(torch.flip(x, dims=[3]))
                logits = (logits1 + logits2) / 2
            else:
                logits = model(x)

            # Top-1
            pred = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

            # Top-5
            k = min(5, logits.shape[1])
            topk = torch.topk(logits, k=k, dim=1).indices
            top5_correct += (topk.eq(y.view(-1,1))).any(dim=1).sum().item()

            n += y.size(0)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    top1_acc = (y_true == y_pred).mean()
    top5_acc = top5_correct / n
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"\n== TEST SONUÇLARI ==")
    print(f"Top-1 Accuracy : {top1_acc*100:.2f}%")
    print(f"Top-5 Accuracy : {top5_acc*100:.2f}%")
    print(f"Macro F1       : {f1_macro*100:.2f}%")
    print(f"Num samples    : {n}")
    print(f"Model          : {args.model_name}  |  Cihaz: {dev}")

if __name__ == "__main__":
    main()
