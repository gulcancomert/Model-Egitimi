
# Yazlab 3 - MultiZoo Transformer Projesi (DeiT + PySide6 Arayüz)

## Adımlar
1. **Sanal ortam oluştur ve kütüphaneleri yükle**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

pip install --upgrade pip
pip install -r requirements.txt
```

2. **Veri setini yerleştir**
```
data/
  train/
  test/
```
Google Drive'dan gelen `train` ve `test` klasörlerini buraya kopyala.

3. **Veri analizi**
```bash
python utils/dataset_stats.py --data-root data/train
```

4. **Doğrulama seti ayır**
```bash
python split_val.py --train-root data/train --val-root data/val --val-ratio 0.2
```

5. **Eğitim (DeiT)**
```bash
python train.py --train-root data/train --val-root data/val --num-classes 90   --model deit_tiny_patch16_224 --batch-size 32 --epochs 30 --lr 1e-4
```

6. **Test setinde değerlendirme**
```bash
python evaluate.py --data-root data/test --checkpoint outputs/checkpoints/best.pth --num-classes 90
```

7. **Arayüzü çalıştır**
```bash
python gui.py --checkpoint outputs/checkpoints/best.pth --class-map outputs/checkpoints/class_map.json   --model deit_tiny_patch16_224 --image-size 224
```
