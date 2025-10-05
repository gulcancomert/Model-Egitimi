

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  

import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import dearpygui.dearpygui as dpg

# --- Sizin core kodlarınız ---
import torch
from core.model_loader import ModelBundle
from core.preprocess import build_infer_transform


# ======================
#   GLOBAL DURUM
# ======================
STATE = {
    "image_path": None,
    "texture_id": None,
    "texture_w": 1,
    "texture_h": 1,
    "temperature": 0.7,
    "use_attention": False,   
    "busy": False,
}

TOP5_IDS: List[Tuple[int, int, int]] = []  # (label_id, bar_id, percent_id)
LOG_LIST_ID = None
TEX_REG_ID = None
MAIN_DRAWLIST_ID = None
MAIN_IMAGE_NODE_ID = None


#MODELİ TEK SEFER YÜKLE

ROOT = Path(__file__).resolve().parents[1]  # repo kökü
MODEL_PATH = ROOT / "outputs_deit" / "best.pt"
CLASSES_PATH = ROOT / "app" / "data" / "classes.json"

BUNDLE = None
TFM = None
AMP = False

def ensure_model():
    global BUNDLE, TFM, AMP
    if BUNDLE is not None:
        return
    # Model + sınıfları yükle
    BUNDLE = ModelBundle(
        model_path=str(MODEL_PATH),
        classes_path=str(CLASSES_PATH),
        device_preference="auto",
        model_name="deit_small_patch16_224"
    ).load()
    TFM = build_infer_transform(224, normalize=True)
    AMP = (getattr(BUNDLE.device, "type", str(BUNDLE.device)) == "cuda")
    log(f"Model hazır: {BUNDLE.model_name if hasattr(BUNDLE,'model_name') else 'DeiT'} / {BUNDLE.device} / {len(BUNDLE.classes)} sınıf")


# ======================
#   YARDIMCILAR
# ======================
def log(msg: str):
    dpg.add_text(msg, parent=LOG_LIST_ID)
    dpg.set_y_scroll(LOG_LIST_ID, 1e9)  # auto-scroll


def load_image_to_texture(path: str):
    """Görseli RGBA'ya çevir ve dinamik texture'a bas."""
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    tex_data = (np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32) / 255.0).reshape(h * w * 4)

    # eski texture varsa temizle
    if STATE["texture_id"] and dpg.does_item_exist(STATE["texture_id"]):
        dpg.delete_item(STATE["texture_id"])

    tid = dpg.add_dynamic_texture(w, h, tex_data, parent=TEX_REG_ID)
    STATE["texture_id"] = tid
    STATE["texture_w"] = w
    STATE["texture_h"] = h

    # drawlist'i temizle ve resmi yerleştir
    dpg.delete_item(MAIN_IMAGE_NODE_ID, children_only=True)
    dpg.draw_image(tid, pmin=(0, 0), pmax=(w, h), parent=MAIN_IMAGE_NODE_ID)
    dpg.configure_item(MAIN_DRAWLIST_ID, width=min(900, w), height=min(700, h))


def file_dialog_callback(sender, app_data):
    sel = app_data.get("file_path_name", "")
    if not sel:
        return
    STATE["image_path"] = sel
    load_image_to_texture(sel)
    log(f"Görsel yüklendi: {sel}")


def open_image_menu():
    dpg.show_item("file_dialog")


def ui_set_temperature(sender, app_data):
    STATE["temperature"] = float(app_data)


def ui_set_attention(sender, app_data):
    STATE["use_attention"] = bool(app_data)


def real_inference(path: str, T: float) -> Tuple[List[str], List[float]]:
    """Sizin eğitimli modelinizle top-5 döndürür."""
    ensure_model()
    img = Image.open(path).convert("RGB")
    x = TFM(img).unsqueeze(0).to(BUNDLE.device)

    T = max(float(T), 1e-6)  # 0 bölme koruması
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=AMP):
            logits = BUNDLE.model(x)               # [1, C]
    probs = torch.softmax(logits / T, dim=1).cpu().numpy()[0]

    topk = min(5, len(BUNDLE.classes))
    idx = np.argsort(-probs)[:topk]
    names = [BUNDLE.classes[i] for i in idx]
    vals  = [float(probs[i]) for i in idx]
    return names, vals


def update_top5(names: List[str], probs: List[float]):
    for i in range(5):
        lbl_id, bar_id, pct_id = TOP5_IDS[i]
        if i < len(names):
            dpg.set_value(lbl_id, f"{i+1}. {names[i]}")
            dpg.set_value(bar_id, probs[i])
            dpg.set_value(pct_id, f"%{probs[i]*100:.2f}")
        else:
            dpg.set_value(lbl_id, f"{i+1}. -")
            dpg.set_value(bar_id, 0.0)
            dpg.set_value(pct_id, "%0.0")


def run_inference_threaded():
    if STATE["busy"]:
        return
    if not STATE["image_path"]:
        log("Önce bir görsel seçin.")
        return

    def _job():
        try:
            STATE["busy"] = True
            log("Tahmin başlıyor...")
            names, probs = real_inference(STATE["image_path"], STATE["temperature"])

            dpg.split_frame()
            update_top5(names, probs)
            top1 = probs[0] if probs else 0.0
            top2 = probs[1] if len(probs) > 1 else 0.0
            dpg.set_value("top1_label", f"Top-1: %{top1*100:.1f}   |   Fark: {top1-top2:.2f}")
            log("Tahmin tamamlandı.")
        except Exception as ex:
            log(f"Hata: {ex}")
        finally:
            STATE["busy"] = False

    threading.Thread(target=_job, daemon=True).start()



#Uİ oluşturuyorum

dpg.create_context()
dpg.create_viewport(title="Zoo Classifier (Dear PyGui)", width=1200, height=820)

with dpg.font_registry():
    default_font = dpg.add_font("C:/Windows/Fonts/Arial.ttf", 20)
dpg.bind_font(default_font)


# Texture registry
TEX_REG_ID = dpg.add_texture_registry()

with dpg.window(label="", tag="root", no_title_bar=True, no_move=True, no_scrollbar=True):
    # Üst bar
    with dpg.group(horizontal=True):
        dpg.add_button(label="Resim Aç", callback=open_image_menu)
        dpg.add_button(label="Tahmin Et", callback=run_inference_threaded)
        dpg.add_checkbox(label="Grad-CAM", callback=ui_set_attention)
        dpg.add_slider_float(label="Temperature", width=220, default_value=STATE["temperature"],
                             min_value=0.1, max_value=3.0, callback=ui_set_temperature)

    dpg.add_spacer(height=6)

    with dpg.group(horizontal=True):
        # Sol: görsel alanı
        with dpg.child_window(width=860, height=680, border=True):
            with dpg.drawlist(width=820, height=640) as MAIN_DRAWLIST_ID:
                dpg.draw_layer()
                with dpg.draw_node() as MAIN_IMAGE_NODE_ID:
                    pass

        # Sağ: Top-5
        with dpg.child_window(width=300, height=680, border=True):
            dpg.add_text("Top-5 Tahmin")
            dpg.add_spacer(height=6)
            for i in range(5):
                lbl = dpg.add_text(f"{i+1}. -")
                pb = dpg.add_progress_bar(default_value=0.0, overlay="", width=-1, height=18)
                pct = dpg.add_text("%0.0")
                dpg.add_spacer(height=4)
                TOP5_IDS.append((lbl, pb, pct))

            dpg.add_separator()
            dpg.add_text("", tag="top1_label")

    dpg.add_spacer(height=6)
    with dpg.child_window(height=110, border=True) as LOG_LIST_ID:
        dpg.add_text("Log:")

# Dosya seçici
with dpg.file_dialog(directory_selector=False, show=False, callback=file_dialog_callback,
                     id="file_dialog", width=700, height=400):
    dpg.add_file_extension(".png,.jpg,.jpeg,.bmp,.webp")
    dpg.add_file_extension(".*")

# Koyu tema
with dpg.theme() as dark_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (20,20,20))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (230,230,230))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (60,60,60))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
dpg.bind_theme(dark_theme)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("root", True)
dpg.start_dearpygui()
dpg.destroy_context()
