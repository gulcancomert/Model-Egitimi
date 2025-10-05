
import torch, timm, json

CKPT = "outputs_deit/best.pt"    
CLASSES = "app/data/classes.json"

with open(CLASSES, "r", encoding="utf-8") as f:
    mp = json.load(f)
num_classes = len(mp)

cands = [
    "deit_small_patch16_224",
    "deit_tiny_patch16_224",
    "deit_base_patch16_224",
]
state = torch.load(CKPT, map_location="cpu")
if isinstance(state, dict) and "model" in state:
    state = state["model"]

best = None
for name in cands:
    m = timm.create_model(name, pretrained=False, num_classes=num_classes)
    res = m.load_state_dict(state, strict=False)
    miss = len(res.missing_keys)
    unexp = len(res.unexpected_keys)
    head_out = getattr(getattr(m, "head", None), "out_features", None)
    score = miss + unexp + (0 if head_out == num_classes else 9999)
    print(f"{name:25s} -> missing={miss:3d}  unexpected={unexp:3d}  head_out={head_out}")
    if best is None or score < best[0]:
        best = (score, name)

print("\nEN UYUMLU:", best[1])
