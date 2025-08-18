# core/model_loader.py
import json, torch, timm

class ModelBundle:
    def __init__(self, model_path, classes_path, device_preference="auto", model_name="deit_small_patch16_224"):
        self.model_path = model_path
        self.classes_path = classes_path
        self.device = torch.device("cuda" if (device_preference in ["auto","cuda"] and torch.cuda.is_available()) else "cpu")
        self.model_name = model_name
        self.model = None
        self.classes = None

    def load(self):
        # 1) sınıf isimleri
        with open(self.classes_path, "r", encoding="utf-8") as f:
            cls_map = json.load(f)
        if isinstance(next(iter(cls_map.keys())), str):
            ordered = [cls_map[str(i)] for i in range(len(cls_map))]
        else:
            ordered = cls_map
        self.classes = ordered
        num_classes = len(self.classes)

        # 2) model iskeleti: AD = deit_small_patch16_224
        model = timm.create_model(self.model_name, pretrained=False, num_classes=num_classes)

        # 3) checkpoint yükle (train_deit.py 'model' anahtarında kaydetti)
        state = torch.load(self.model_path, map_location="cpu")  # uyarı normal; dosya güvenli, biz kaydettik
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # 4) yükle ve kontrol et
        res = model.load_state_dict(state, strict=False)
        head_out = getattr(getattr(model, "head", None), "out_features", None)
        if head_out != num_classes:
            raise RuntimeError(f"Head out_features {head_out} != num_classes {num_classes}")

        model.to(self.device).eval()
        self.model = model
        return self
