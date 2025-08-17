import os, json, csv, math
from typing import List
import numpy as np
from PIL import Image
from io import BytesIO

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QProgressBar, QGroupBox, QListWidget, QListWidgetItem,
    QSplitter, QScrollArea, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QFormLayout, QSpinBox, QComboBox, QLineEdit, QDialog, QDialogButtonBox,
    QStatusBar, QToolBar
)

import torch

try:
    import qdarktheme
    HAS_QDARK = True
except Exception:
    HAS_QDARK = False

from core.model_loader import ModelBundle
from core.inference_worker import ImageInferenceWorker
from core.preprocess import build_infer_transform

from utils.logger import UILogger


def np_to_qimage(arr: np.ndarray) -> QImage:
    # expects HxWxC in [0,255]
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    h, w, c = arr.shape
    bytes_per_line = c * w
    qimg = QImage(arr.astype(np.uint8).data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return qimg.copy()

def overlay_heatmap(base_img: Image.Image, heat: np.ndarray, alpha=0.45):
    import cv2
    # base_img: PIL RGB, heat: 0..1 float
    base = np.array(base_img.convert("RGB"))
    hm = (heat * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    overlay = (alpha * hm + (1 - alpha) * base).clip(0,255).astype(np.uint8)
    return Image.fromarray(overlay)

class SettingsDialog(QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("Ayarlar")
        self.config = config or {}
        form = QFormLayout(self)

        self.device_box = QComboBox()
        self.device_box.addItems(["auto", "cpu", "cuda"])
        self.device_box.setCurrentText(self.config.get("device", "auto"))

        self.img_size = QSpinBox()
        self.img_size.setRange(96, 1024)
        self.img_size.setValue(self.config.get("image_size", 224))

        self.model_name = QLineEdit(self.config.get("model_name", "deit_small_patch16_224"))
        self.model_path = QLineEdit(self.config.get("model_path", "outputs_deit/best.pt"))
        self.class_path = QLineEdit(self.config.get("classes_path", "app/data/classes.json"))

        self.normalize_chk = QCheckBox("Normalize (ImageNet mean/std)")
        self.normalize_chk.setChecked(self.config.get("normalize", True))

        self.gradcam_chk = QCheckBox("Attention Rollout (Grad-CAM)")
        self.gradcam_chk.setChecked(self.config.get("use_attention", False))

        form.addRow("Cihaz", self.device_box)
        form.addRow("Görüntü boyutu", self.img_size)
        form.addRow("Model adı", self.model_name)
        form.addRow("Model dosyası", self.model_path)
        form.addRow("Sınıf dosyası", self.class_path)
        form.addRow(self.normalize_chk)
        form.addRow(self.gradcam_chk)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def values(self):
        return {
            "device": self.device_box.currentText(),
            "image_size": self.img_size.value(),
            "model_name": self.model_name.text().strip(),
            "model_path": self.model_path.text().strip(),
            "classes_path": self.class_path.text().strip(),
            "normalize": self.normalize_chk.isChecked(),
            "use_attention": self.gradcam_chk.isChecked(),
        }

class ImagePreview(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumSize(320, 320)
        self._pix = None
        self._scale = 1.0
        self.setStyleSheet("QLabel { background: #222; border: 1px solid #444; }")
        self.setText("\n\n Görsel sürükleyip bırakın")
        
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.toLocalFile().lower().endswith((".png",".jpg",".jpeg",".bmp")):
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            path = u.toLocalFile()
            if path.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                mw = self.window()  # QMainWindow
                if hasattr(mw, "load_image_path"):
                    mw.load_image_path(path)
                break
        

    def wheelEvent(self, e):
        if self._pix:
            delta = e.angleDelta().y()
            factor = 1.2 if delta > 0 else 1/1.2
            self._scale = float(np.clip(self._scale * factor, 0.2, 5.0))
            self._update_scaled()

    def set_image(self, qpix: QPixmap):
        self._pix = qpix
        self._scale = 1.0
        self._update_scaled()

    def _update_scaled(self):
        if not self._pix:
            return
        w = int(self._pix.width() * self._scale)
        h = int(self._pix.height() * self._scale)
        self.setPixmap(self._pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zoo Classifier (DeiT)")
        self.setMinimumSize(1100, 700)
        self.logger = UILogger()
        self.logger.message.connect(self._append_log)

        # Default config
        self.config = {
            "device": "auto",
            "image_size": 224,
            "normalize": True,
            "use_attention": False,
            "model_name": "deit_small_patch16_224",
            "model_path": "outputs_deit/best.pt",
            "classes_path": "app/data/classes.json"
        }

        # Toolbar
        tb = QToolBar("Araçlar")
        self.addToolBar(tb)
        act_open = QAction("Resim Aç", self)
        act_open.triggered.connect(self.open_image)
        tb.addAction(act_open)

        act_folder = QAction("Klasör Tahmini", self)
        act_folder.triggered.connect(self.batch_folder)
        tb.addAction(act_folder)

        act_export = QAction("CSV Aktar", self)
        act_export.triggered.connect(self.export_csv)
        tb.addAction(act_export)

        act_settings = QAction("Ayarlar", self)
        act_settings.triggered.connect(self.open_settings)
        tb.addAction(act_settings)

        act_theme = QAction("Tema", self)
        act_theme.triggered.connect(self.toggle_theme)
        tb.addAction(act_theme)

        # Splitter: left preview, right results
        splitter = QSplitter(self)

        # Left: image preview
        left = QWidget()
        left_lay = QVBoxLayout(left)
        self.preview = ImagePreview()
        left_lay.addWidget(self.preview)

        btns = QWidget()
        btn_lay = QHBoxLayout(btns)
        self.btn_predict = QPushButton("Tahmin Et")
        self.btn_predict.clicked.connect(self.predict_current)
        self.chk_attention = QCheckBox("Grad-CAM (attention)")
        self.chk_attention.setChecked(False)
        btn_lay.addWidget(self.btn_predict)
        btn_lay.addWidget(self.chk_attention)
        left_lay.addWidget(btns)

        splitter.addWidget(left)

        # Right: results
        right = QWidget()
        right_lay = QVBoxLayout(right)

        # Top-5 list
        self.top5_group = QGroupBox("Top-5 Tahmin")
        g_lay = QVBoxLayout(self.top5_group)
        self.rows = []
        for i in range(5):
            row = QWidget()
            rlay = QHBoxLayout(row)
            lbl = QLabel(f"{i+1}. sınıf")
            bar = QProgressBar()
            bar.setRange(0, 100)
            val = QLabel("%0.00")
            rlay.addWidget(lbl, 2)
            rlay.addWidget(bar, 6)
            rlay.addWidget(val, 2)
            g_lay.addWidget(row)
            self.rows.append((lbl, bar, val))
        right_lay.addWidget(self.top5_group)

        # Meta table
        self.meta = QTableWidget(3, 2)
        self.meta.setHorizontalHeaderLabels(["Özellik", "Değer"])
        self.meta.verticalHeader().setVisible(False)
        self.meta.horizontalHeader().setStretchLastSection(True)
        self.meta.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.meta.setItem(0,0, QTableWidgetItem("Görsel boyutu"))
        self.meta.setItem(1,0, QTableWidgetItem("Cihaz / süre (ms)"))
        self.meta.setItem(2,0, QTableWidgetItem("Dosya"))
        right_lay.addWidget(self.meta)

        splitter.addWidget(right)
        splitter.setSizes([600, 500])

        # Log panel
        self.log = QListWidget()
        right_lay.addWidget(QLabel("Log"))
        right_lay.addWidget(self.log)

        # Central
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(splitter)
        self.setCentralWidget(central)

        # Status bar
        self.setStatusBar(QStatusBar())

        # Model bundle (lazy load)
        self.bundle = None

        # Data holders
        self.current_image_path = None
        self.batch_results = []

    # THEME
    def toggle_theme(self):
        if HAS_QDARK:
            # Eğer daha önce ayarlanmamışsa varsayılan light yap
            self._theme = getattr(self, "_theme", "light")
            # light ↔ dark geçişi
            self._theme = "dark" if self._theme == "light" else "light"
            qdarktheme.setup_theme(self._theme)
        else:
            QMessageBox.information(self, "Tema", "qdarktheme kurulu değil.")
        
        
    # SETTINGS
    def open_settings(self):
        dlg = SettingsDialog(self, self.config)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.config = dlg.values()
            self._append_log("Ayarlar güncellendi.")
            # reload model lazily next predict

    # OPEN IMAGE
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Görseller (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.load_image_path(path)

    def load_image_path(self, path: str):
        self.current_image_path = path
        img = Image.open(path).convert("RGB")
        qimg = QImage(img.tobytes(), img.size[0], img.size[1], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview.set_image(pix)
        self.meta.setItem(0,1, QTableWidgetItem(f"{img.size[0]}x{img.size[1]}"))
        self.meta.setItem(2,1, QTableWidgetItem(os.path.basename(path)))
        self._append_log(f"Görsel yüklendi: {path}")

    # BATCH FOLDER
    def batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Klasör Seç")
        if not folder:
            return
        exts = (".jpg",".jpeg",".png",".bmp")
        paths = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        if not paths:
            QMessageBox.warning(self, "Uyarı", "Klasörde uygun görsel bulunamadı.")
            return
        self._append_log(f"{len(paths)} görsel için toplu tahmin başlıyor...")
        self.run_inference(paths, batch=True)

    # EXPORT CSV
    def export_csv(self):
        if not self.batch_results:
            QMessageBox.information(self, "Bilgi", "Önce toplu tahmin yapın.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "CSV Kaydet", "results.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file","top1_class","top1_prob","latency_ms"] + [f"top{i}_class" for i in range(2,6)] + [f"top{i}_prob" for i in range(2,6)])
            for r in self.batch_results:
                top_idx = r.topk_idx
                top_prob = r.topk_prob
                names = [self.bundle.classes[i] for i in top_idx]
                row = [r.path, names[0], f"{top_prob[0]:.4f}", f"{r.latency_ms:.2f}"]
                # rest
                row += names[1:] + [f"{p:.4f}" for p in top_prob[1:]]
                w.writerow(row)
        self._append_log(f"CSV kaydedildi: {path}")
        QMessageBox.information(self, "CSV", "CSV başarıyla kaydedildi.")

    # PREDICT
    def predict_current(self):
        if not self.current_image_path:
            QMessageBox.information(self, "Bilgi", "Önce bir görsel seçin.")
            return
        self.run_inference([self.current_image_path], batch=False)

    # INFERENCE
    def ensure_model(self):
        if self.bundle is not None:
            return
        cfg = self.config
        if not os.path.exists(cfg["model_path"]):
            QMessageBox.critical(self, "Model bulunamadı", f"Model dosyası yok: {cfg['model_path']}")
            return
        if not os.path.exists(cfg["classes_path"]):
            QMessageBox.critical(self, "Sınıf dosyası yok", f"Sınıf dosyası: {cfg['classes_path']}")
            return
        self._append_log("Model yükleniyor...")
        self.bundle = ModelBundle(cfg["model_path"], cfg["classes_path"],
                                  device_preference=cfg["device"],
                                  model_name=cfg["model_name"]).load()
        self._append_log(f"Model hazır: {cfg['model_name']} / {self.bundle.device} / {len(self.bundle.classes)} sınıf")

    def run_inference(self, paths: List[str], batch: bool):
        self.ensure_model()
        if not self.bundle:
            return
        cfg = self.config
        self.worker = ImageInferenceWorker(self.bundle, device=cfg["device"],
                                           image_size=cfg["image_size"],
                                           normalize=cfg["normalize"],
                                           use_attention=self.chk_attention.isChecked(),
                                           batch_paths=paths)
        self.worker.result_ready.connect(self._on_result)
        self.worker.batch_done.connect(self._on_batch_done)
        self.worker.error.connect(self._on_error)
        self.worker.progress.connect(self._on_progress)
        self.worker.start()

    # RESULT HANDLERS
    def _on_result(self, res):
        # Update top-5
        names = [self.bundle.classes[i] for i in res.topk_idx]
        probs = res.topk_prob
        for i, (lbl, bar, val) in enumerate(self.rows):
            if i < len(names):
                lbl.setText(f"{i+1}. {names[i]}")
                bar.setValue(int(probs[i] * 100))
                val.setText(f"%{probs[i]*100:.2f}")
            else:
                lbl.setText(f"{i+1}. -")
                bar.setValue(0)
                val.setText("%0.00")
        # meta
        dev = getattr(self.bundle.device, "type", str(self.bundle.device))
        self.meta.setItem(1,1, QTableWidgetItem(f"{dev} / {res.latency_ms:.2f}"))
        # attention overlay if available
        if res.attn_map is not None and self.current_image_path and os.path.samefile(res.path, self.current_image_path):
            base = Image.open(self.current_image_path).convert("RGB")
            over = overlay_heatmap(base, res.attn_map, alpha=0.45)
            qimg = QImage(over.tobytes(), over.size[0], over.size[1], QImage.Format.Format_RGB888)
            self.preview.set_image(QPixmap.fromImage(qimg))

        self.batch_results.append(res)

    def _on_batch_done(self, results):
        self._append_log(f"Tahmin tamamlandı: {len(results)} görsel.")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Hata", msg)
        self._append_log(f"Hata: {msg}")

    def _on_progress(self, p):
        self.statusBar().showMessage(f"İlerleme: %{p}")

    # LOG
    def _append_log(self, text: str):
        self.log.addItem(text)
        self.log.scrollToBottom()

def main():
    import sys
    app = QApplication(sys.argv)
    if HAS_QDARK:
        qdarktheme.setup_theme("dark")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
