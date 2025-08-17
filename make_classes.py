import json, os

# train klasörünün yolu
p = "data/train"

# alfabetik sırayla klasörleri al
cls = sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])

# app/data klasörünü oluştur (varsa sorun yok)
os.makedirs("app/data", exist_ok=True)

# id -> class eşleşmesini json'a yaz
with open("app/data/classes.json", "w", encoding="utf-8") as f:
    json.dump({str(i): c for i, c in enumerate(cls)}, f, ensure_ascii=False, indent=2)

print("classes.json oluşturuldu ✅")
