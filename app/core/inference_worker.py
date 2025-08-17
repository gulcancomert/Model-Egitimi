import os, time, csv
from typing import List, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from PySide6.QtCore import QThread, Signal

from .preprocess import build_infer_transform
from .gradcam import VitAttentionRollout, attention_to_map

def load_image(path: str):
    img = Image.open(path).convert("RGB")
    return img

class InferenceResult:
    def __init__(self, path: str, topk_idx, topk_prob, latency_ms: float, attn_map=None):
        self.path = path
        self.topk_idx = topk_idx
        self.topk_prob = topk_prob
        self.latency_ms = latency_ms
        self.attn_map = attn_map

class ImageInferenceWorker(QThread):
    result_ready = Signal(object)    # InferenceResult
    batch_done = Signal(list)        # list of InferenceResult
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, model_bundle, device: str = "auto", image_size=224, normalize=True,
                 use_attention=False, batch_paths: Optional[List[str]]=None, parent=None):
        super().__init__(parent)
        self.bundle = model_bundle
        self.device_pref = device
        self.image_size = image_size
        self.normalize = normalize
        self.use_attention = use_attention
        self.batch_paths = batch_paths or []
        self._stop = False

    def stop(self):
        self._stop = True


    def run(self):
        try:
            device = self.bundle.device          # torch.device('cuda') / 'cpu'
            model  = self.bundle.model
            classes = self.bundle.classes

            # 1) Modeli eval moduna al (KRİTİK)
            model.eval()

            # 2) Preprocess (eğitimle bire bir aynı)
            tfm = build_infer_transform(self.image_size, self.normalize)

            # 3) (İsteğe bağlı) Attention rollout
            rollout = VitAttentionRollout(model) if self.use_attention else None

            results = []
            paths = self.batch_paths if self.batch_paths else [None]
            total = len(paths)

            # 4) Autocast + inference_mode ile hızlı ve güvenli inference
            amp_enabled = (str(device) == "cuda")

            for idx, p in enumerate(paths):
                if self._stop:
                    break
                if p is None:
                    self.error.emit("No image path provided.")
                    return

                img = load_image(p)
                w, h = img.size
                x = tfm(img).unsqueeze(0).to(device)

                if rollout:
                    rollout.clear()

                t0 = time.perf_counter()
                with torch.inference_mode():
                    # (opsiyonel) cuda'da kararlı hız için autocast
                    with torch.autocast("cuda", enabled=amp_enabled):
                        logits = model(x)              # [1, num_classes]
                elapsed = (time.perf_counter() - t0) * 1000.0

                # 5) Softmax -> Top-5
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                topk = min(5, len(classes))
                topk_idx  = np.argsort(-probs)[:topk]
                topk_prob = probs[topk_idx]

                # 6) (İsteğe bağlı) Attention heatmap
                attn_map = None
                if rollout:
                    mask = rollout()                   # [L, L] ya da benzeri
                    P = mask.shape[-1]
                    g = int(P ** 0.5)
                    maps = attention_to_map(mask, g, (h, w))
                    attn_map = maps[0]

                res = InferenceResult(p, topk_idx, topk_prob, elapsed, attn_map)
                results.append(res)
                self.result_ready.emit(res)
                self.progress.emit(int(100 * (idx + 1) / total))

            self.batch_done.emit(results)

        except Exception as e:
            self.error.emit(str(e))
