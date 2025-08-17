
import argparse, os, collections

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def is_img(p):
    return os.path.splitext(p.lower())[1] in IMG_EXTS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    class_counts = collections.Counter()
    total = 0
    classes = []
    for cls in sorted(os.listdir(args.data_root)):
        cdir = os.path.join(args.data_root, cls)
        if not os.path.isdir(cdir):
            continue
        classes.append(cls)
        cnt = 0
        for fname in os.listdir(cdir):
            if is_img(fname):
                cnt += 1
        class_counts[cls] = cnt
        total += cnt

    print(f"Kök klasör: {args.data_root}")
    print(f"Sınıf sayısı: {len(classes)}")
    print(f"Toplam görsel: {total}")
    if total > 0:
        avg = total / max(1, len(classes))
        print(f"Ortalama/sınıf: {avg:.2f}")

    print("\n--- Sınıf başına adet ---")
    for cls in classes:
        print(f"{cls}: {class_counts[cls]}")

if __name__ == "__main__":
    main()
