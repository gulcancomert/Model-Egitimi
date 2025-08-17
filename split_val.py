
import argparse, os, shutil, random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def is_img(p):
    return Path(p).suffix.lower() in IMG_EXTS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.val_root, exist_ok=True)

    classes = [d for d in os.listdir(args.train_root) if os.path.isdir(os.path.join(args.train_root, d))]
    for cls in classes:
        src_dir = os.path.join(args.train_root, cls)
        dst_dir = os.path.join(args.val_root, cls)
        os.makedirs(dst_dir, exist_ok=True)

        imgs = [f for f in os.listdir(src_dir) if is_img(f)]
        if len(imgs) < 5:
            print(f"[UYARI] {cls} sınıfında az görsel var ({len(imgs)}).")
        random.shuffle(imgs)
        k = max(1, int(len(imgs) * args.val_ratio))
        val_files = imgs[:k]

        for f in val_files:
            shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

        print(f"{cls}: {len(val_files)} görsel doğrulamaya taşındı.")

if __name__ == "__main__":
    main()
