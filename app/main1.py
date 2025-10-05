
import os, csv, tempfile, traceback
from typing import List
import numpy as np
from PIL import Image

from PySide6.QtCore import QObject, Slot, Signal, QUrl, QThread
from PySide6.QtCore import QTimer
from core.model_loader import ModelBundle
from core.inference_worker import ImageInferenceWorker
from utils.logger import UILogger


def _overlay_heatmap(base_img: Image.Image, heat: np.ndarray, alpha=0.45):
    """base_img: PIL RGB, heat: 0..1 float"""
    try:
        import cv2
        base = np.array(base_img.convert("RGB"))
        hm = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        blend = (alpha * hm + (1 - alpha) * base).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blend)
    except Exception:
        # cv2 yoksa basit kırmızı overlay
        heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
        r = heat_u8
        g = np.zeros_like(r)
        b = np.zeros_like(r)
        hm = np.stack([r, g, b], axis=-1)
        base = np.array(base_img.convert("RGB"), dtype=np.uint8)
        blend = (alpha * hm + (1 - alpha) * base).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blend)


class InferenceThread(QThread):
    """ImageInferenceWorker'ı arka planda çalıştırır."""
    result_ready = Signal(object)   # res nesnesi
    batch_done = Signal(list)       # sonuç listesi
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, bundle: ModelBundle, paths: List[str], cfg: dict, use_attention: bool):
        super().__init__()
        self.bundle = bundle
        self.paths = paths
        self.cfg = cfg
        self.use_attention = use_attention
        self.worker = None
        self._last_size = "-"   # ✅ eklendi
        self._last_device_latency = "-"
        self._last_file_name = "-"

    def _emit_meta(self):
        """Son değerleri QML'e gönderir."""
        self.metaChanged.emit(self._last_size, self._last_device_latency, self._last_file_name)
    
    
    def run(self):
        try:
            self.worker = ImageInferenceWorker(
                self.bundle,
                device=self.cfg["device"],
                image_size=self.cfg["image_size"],
                normalize=self.cfg["normalize"],
                use_attention=self.use_attention,
                batch_paths=self.paths,
                temperature=self.cfg["temperature"],
            )
            self.worker.result_ready.connect(self.result_ready)
            self.worker.batch_done.connect(self.batch_done)
            self.worker.error.connect(self.error)
            self.worker.progress.connect(self.progress)
            self.worker.run()  # Worker kendi thread'inde çalışıyor (bloklamaz)
        except Exception as e:
            self.error.emit(f"Inference error: {e}\n{traceback.format_exc()}")

    def stop(self):
        try:
            if self.worker:
                self.worker.stop()
        except Exception:
            pass


class Backend(QObject):
    # QML bağları
    imageChanged = Signal(str)       
    resultsChanged = Signal(list)     
    busyChanged = Signal(bool)
    messageChanged = Signal(str)
    progressChanged = Signal(int)
    metaChanged = Signal(str, str, str)  # size_text, device_latency, file_name
    logAdded = Signal(str)
    infoChanged = Signal(str)  #yeni sinyal

    def __init__(self):
        super().__init__()
        self.logger = UILogger()
        self.logger.message.connect(self._append_log)

        # Config (ilk arayüzle aynı alanlar)
        self.cfg = {
            "device": "auto",
            "image_size": 224,
            "normalize": True,
            "use_attention": False,
            "model_name": "deit_small_patch16_224",
            "model_path": "outputs_deit/best.pt",
            "classes_path": "app/data/classes.json",
            "temperature": 0.7,
        }

        self._image_path: str = ""
        self._busy = False
        self.bundle: ModelBundle | None = None
        self.thread: InferenceThread | None = None
        self.batch_results = []

    # ---------- Helpers ----------
    def _set_busy(self, b: bool):
        self._busy = b
        self.busyChanged.emit(b)

    def _append_log(self, text: str):
        self.logAdded.emit(text)

    def _ensure_model(self) -> bool:
        if self.bundle is not None:
            return True
        if not os.path.exists(self.cfg["model_path"]):
            self.messageChanged.emit(f"Model dosyası yok: {self.cfg['model_path']}")
            return False
        if not os.path.exists(self.cfg["classes_path"]):
            self.messageChanged.emit(f"Sınıf dosyası yok: {self.cfg['classes_path']}")
            return False
        self._append_log("Model yükleniyor...")
        self.bundle = ModelBundle(
            self.cfg["model_path"],
            self.cfg["classes_path"],
            device_preference=self.cfg["device"],
            model_name=self.cfg["model_name"]
        ).load()
        self._append_log(
            f"Model hazır: {self.cfg['model_name']} / {self.bundle.device} / {len(self.bundle.classes)} sınıf"
        )
        return True

    # ---------- Slots (QML çağırır) ----------
    @Slot(str)
    def loadImage(self, fileUrl: str):
        path = QUrl(fileUrl).toLocalFile()
        if not path:
            return
        self._image_path = path
        # Preview için orijinal resmi göster
        self.imageChanged.emit("file:///" + path.replace("\\", "/"))
        try:
            with Image.open(path) as im:
                w, h = im.size
            self._last_size = f"{w}x{h}"
            self._last_file_name = os.path.basename(path)
            self.metaChanged.emit(self._last_size, self._last_device_latency, self._last_file_name)

        except Exception:
            pass
        self.messageChanged.emit(f"Görsel yüklendi: {path}")

    @Slot(float)
    def setTemperature(self, v: float):
        self.cfg["temperature"] = float(v)
        self.messageChanged.emit(f"Temperature: {self.cfg['temperature']}")

    @Slot(bool)
    def setGradcam(self, flag: bool):
        self.cfg["use_attention"] = bool(flag)
        self.messageChanged.emit(f"Grad-CAM: {self.cfg['use_attention']}")

    @Slot()
    def runInference(self):
        if not self._image_path:
            self.messageChanged.emit("Önce bir resim seçin.")
            return
        self._run_paths([self._image_path])

    @Slot(str)
    def runBatchOnFolder(self, folderUrl: str):
        folder = QUrl(folderUrl).toLocalFile()
        if not folder:
            self.messageChanged.emit("Klasör seçilmedi.")
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        if not paths:
            self.messageChanged.emit("Klasörde uygun görsel bulunamadı.")
            return
        self._append_log(f"{len(paths)} görsel için toplu tahmin başlıyor…")
        self._run_paths(paths)

    @Slot(str)
    def exportCsv(self, saveFileUrl: str):
        if not self.batch_results:
            self.messageChanged.emit("Önce toplu tahmin yapın.")
            return
        if not self._ensure_model():
            return
        path = QUrl(saveFileUrl).toLocalFile()
        if not path:
            self.messageChanged.emit("Geçersiz CSV dosya yolu.")
            return
        header = ["file", "top1_class", "top1_prob", "latency_ms"] + \
                 [f"top{i}_class" for i in range(2, 6)] + [f"top{i}_prob" for i in range(2, 6)]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in self.batch_results:
                top_idx = list(r.topk_idx)
                top_prob = list(r.topk_prob)
                names = [self.bundle.classes[i] for i in top_idx]
                while len(names) < 5:
                    names.append("-")
                while len(top_prob) < 5:
                    top_prob.append(0.0)
                row = [r.path, names[0], f"{top_prob[0]:.4f}", f"{r.latency_ms:.2f}"]
                row += names[1:5] + [f"{p:.4f}" for p in top_prob[1:5]]
                w.writerow(row)
        self._append_log(f"CSV kaydedildi: {path}")
        self.messageChanged.emit("CSV başarıyla kaydedildi.")


    def _run_paths(self, paths: List[str]):
        if not self._ensure_model():
            return
        if self._busy:
            self.messageChanged.emit("Zaten çalışıyor…")
            return
        self.batch_results = []
        self.resultsChanged.emit([])
        self._set_busy(True)

        self.thread = InferenceThread(
            bundle=self.bundle,
            paths=paths,
            cfg=self.cfg,
            use_attention=self.cfg["use_attention"]
        )
        self.thread.result_ready.connect(self._on_result)
        self.thread.batch_done.connect(self._on_batch_done)
        self.thread.error.connect(self._on_error)
        self.thread.progress.connect(self._on_progress)
        self.thread.finished.connect(lambda: self._set_busy(False))
        self.thread.start()

    # ---------- Worker callbacks ----------
    def _on_result(self, res):
        # Top-5
        names = [self.bundle.classes[i] for i in res.topk_idx]
        probs = list(res.topk_prob)
        view = [f"{names[i]}: {probs[i]*100:.1f}%" for i in range(min(5, len(names)))]
        self.resultsChanged.emit(view)

        # meta
        dev = getattr(self.bundle.device, "type", str(self.bundle.device))
        self._last_device_latency = f"{dev} / {res.latency_ms:.2f}"
        self._last_file_name = os.path.basename(res.path)
        self.metaChanged.emit(self._last_size, self._last_device_latency, self._last_file_name)


        # Grad-CAM overlay (sadece tekil görsel ve eşleşiyorsa)
        try:
            if res.attn_map is not None and self._image_path and \
               os.path.abspath(res.path) == os.path.abspath(self._image_path):
                base = Image.open(self._image_path).convert("RGB")
                heat = res.attn_map
                if heat.ndim > 2:  # <<< FIX
                    heat = np.mean(heat, axis=0)
                # normalize 0..1
                heat = (heat - float(np.min(heat))) / (float(np.ptp(heat)) + 1e-6)
                over = _overlay_heatmap(base, heat, alpha=0.45)
                tmp = os.path.join(tempfile.gettempdir(), "qml_gradcam_overlay.png")
                over.save(tmp)
                self.imageChanged.emit("file:///" + tmp.replace("\\", "/"))
        except Exception as e:
            self._append_log(f"Grad-CAM overlay hatası: {e}")

        # batch listesine ekle
        self.batch_results.append(res)

    def _on_batch_done(self, results):
        self._append_log(f"Tahmin tamamlandı: {len(results)} görsel.")

    def _on_error(self, msg):
        self._append_log(f"Hata: {msg}")
        self.messageChanged.emit(msg)

    def _on_progress(self, p: int):
        self.progressChanged.emit(int(p))

    # ---------- Temizlik ----------
    @Slot()
    def stop(self):
        try:
            if self.thread and self.thread.isRunning():
                if self.thread.worker:   # <<< FIX
                    self.thread.worker.stop()
                self.thread.quit()
                self.thread.wait(2000)
        except Exception:
            pass

if __name__ == "__main__":
    import sys
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtCore import QTimer

    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Backend sınıfını QML'e register et
    backend = Backend()
    engine.rootContext().setContextProperty("backend", backend)

    # İlk önce SplashPage.qml yüklenecek
    splash_file = os.path.join(os.path.dirname(__file__), "SplashPage.qml")
    engine.load(splash_file)

    if not engine.rootObjects():
        sys.exit(-1)

  
    def load_main():
        engine.clearComponentCache()
        qml_file = os.path.join(os.path.dirname(__file__), "main1.qml")
        engine.load(qml_file)
        if not engine.rootObjects():
            sys.exit(-1)

    QTimer.singleShot(3000, load_main)  # 3000 ms = 3 saniye

    sys.exit(app.exec())
