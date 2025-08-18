import os, time, csv
from typing import List, Optional
import torch
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
                 use_attention=False, batch_paths: Optional[List[str]]=None,
                 temperature: float = 1.0, parent=None):
        super().__init__(parent)
        self.bundle = model_bundle
        self.device_pref = device
        self.image_size = image_size
        self.normalize = normalize
        self.use_attention = use_attention
        self.batch_paths = batch_paths or []
        self.temperature = float(temperature)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # --- Giriş validasyonu ---
            if not self.batch_paths:
                self.error.emit("Görüntü yolu listesi boş (batch_paths).")
                return

            device = self.bundle.device          # torch.device('cuda') / 'cpu' / 'mps'
            model  = self.bundle.model
            classes = self.bundle.classes

            if len(classes) == 0:
                self.error.emit("Sınıf listesi boş.")
                return

            # 1) Modeli eval moduna al (KRİTİK)
            model.eval()

            # 2) Preprocess (eğitimle bire bir aynı)
            tfm = build_infer_transform(self.image_size, self.normalize)

            # 3) (İsteğe bağlı) Attention rollout
            rollout = VitAttentionRollout(model) if self.use_attention else None

            results = []
            paths = list(self.batch_paths)
            total = len(paths)

            # 4) Autocast + inference_mode ile hızlı ve güvenli inference
            dev_type = getattr(device, "type", str(device))  # "cuda" / "cpu" / "mps"
            amp_enabled = (dev_type == "cuda")

            for idx, p in enumerate(paths):
                if self._stop:
                    break

                img = load_image(p)
                w, h = img.size
                x = tfm(img).unsqueeze(0).to(device)

                if rollout:
                    rollout.clear()
                    
                def _tta_batch(x):
                    # Basit ve hızlı TTA: orijinal + yatay çeviri
                    xs = [x, torch.flip(x, dims=[-1])]
                    return xs

                t0 = time.perf_counter()
                with torch.inference_mode():
                    with torch.autocast("cuda", enabled=amp_enabled):
                        logits_list = []
                        for xt in _tta_batch(x):
                            logits_list.append(model(xt))     # [1, C]
                        logits = torch.stack(logits_list, dim=0).mean(dim=0)  # [1, C]
                elapsed = (time.perf_counter() - t0) * 1000.0


                # 5) Softmax -> Top-5
                T = self.temperature if self.temperature and self.temperature > 0 else 1.0
                probs = torch.softmax(logits / T, dim=1).cpu().numpy()[0]
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
                    m0 = maps[0].astype(np.float32)
                    m0 = (m0 - m0.min()) / (m0.ptp() + 1e-6)
                    attn_map = m0

                res = InferenceResult(p, topk_idx, topk_prob, elapsed, attn_map)
                results.append(res)
                self.result_ready.emit(res)
                self.progress.emit(int(100 * (idx + 1) / total))

            # stop edildiyse batch_done göndermeyebilirsin
            if self._stop:
                return

            self.batch_done.emit(results)

        except Exception as e:
            self.error.emit(str(e))
