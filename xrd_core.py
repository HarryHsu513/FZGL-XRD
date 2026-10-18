# -*- coding: utf-8 -*-
"""
XRD Studio — 核心程式（在瀏覽器中由 Pyodide 執行，也可在本機 Python 直接使用）
=========================================================================
同步輻射 (.xye / .xyn) 與 In-house (.txt …) XRD 數據
→ 自動判斷格式 → 換算到 In-house 波長 (Cu Kα1) → 期刊級繪圖 → 匯出圖片與 CSV

轉換邏輯來源：Synchrotron_Convert.py（Amanda, 2024-12-25）
  * convert_two_theta()：原封不動保留。
  * process_file() 的流程「去掉標頭 → 第一欄 2θ 換算 → 刪除第三欄 (esd) → 以逗號輸出 CSV」保留，
    只修正了幾個邊界問題，說明寫在 read_xrd_text() 與 converted_csv_text() 的註解中。

本機批次轉換（不開介面，效果等同原程式）：
    python xrd_core.py 檔案1.xye 檔案2.xye ...      → 在原檔旁輸出 CSV
需求：numpy、matplotlib
"""

import csv
import io
import itertools
import json
import logging
import math
import os
import re
import sys
import traceback
import uuid
import warnings as _warnings
import zipfile
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib import font_manager, ticker  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox, ScaledTranslation, blended_transform_factory  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_ps").setLevel(logging.ERROR)  # EPS 不支援透明度的提示

APP_VERSION = "2.3"

# ═════════════════════════════════════════════════════════════════════════════
# 1. 波長設定與換算（原程式碼，邏輯未更動）
# ═════════════════════════════════════════════════════════════════════════════
ORI_WL = 0.77491  # 原始波長
TAG_WL = 1.540598  # 目標波長


def convert_two_theta(two_theta1, lambda1, lambda2):
    """
    將 2𝜃 值從一個波長轉換為另一個波長。

    Args:
        two_theta1 (float): 原始 2𝜃 值（度數）。
        lambda1 (float): 原始波長。
        lambda2 (float): 目標波長。

    Returns:
        float: 新波長下的 2𝜃 值（度數）。
    """
    theta1 = np.radians(two_theta1 / 2)  # 轉換為弧度
    sin_theta2 = np.sin(theta1) * (lambda2 / lambda1)  # 調整波長比例
    theta2 = np.arcsin(sin_theta2)  # 計算新的 θ 值
    return np.degrees(theta2 * 2)  # 返回 2𝜃（度數）


# ═════════════════════════════════════════════════════════════════════════════
# 2. 預設值（= Nature 單欄樣式）、期刊樣式、配色
# ═════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    # 波長（只有在 .xye 沒記載波長時才會用到）
    "src_wl": ORI_WL,
    # 數據處理
    "smooth": "none", "smooth_win": 7, "smooth_order": 2,
    "bg": "none", "bg_width": 1.5,
    "transform": "linear",
    "norm": "max", "norm_scope": "view", "norm_x1": None, "norm_x2": None,
    # 排列
    "mode": "stack", "stack_method": "equal", "stack_spacing": 1.1, "stack_gap": 0.08,
    "first_on_top": True,
    # 線條與顏色
    "palette": "okabe", "lw": 0.75, "fill": False, "fill_alpha": 0.15,
    # 曲線名稱與圖例
    "label_mode": "auto", "clabel_pos": "right", "clabel_pad": 0.06, "clabel_color": True,
    "legend_loc": "upper right", "legend_frame": False, "legend_ncol": 1, "legend_handle": 1.6,
    # X / Y 軸
    "xlabel": "2θ (°)", "ylabel": "Intensity (a.u.)",
    "xmin": None, "xmax": None, "x_major": None, "x_minor": 2,
    "y_ticks": False, "y_ticklabels": False, "ymargin_top": 0.06, "ymargin_bottom": 0.03,
    # 外框與刻度
    "frame": "box", "tick_dir": "in", "tick_top": False, "tick_right": False,
    "tick_major_len": 3.0, "tick_minor_len": 1.6, "tick_width": 0.5, "spine_lw": 0.5,
    "grid": "none", "sec_axis": "none",
    # 文字
    "font": "Arial", "fs_label": 7, "fs_tick": 6, "fs_legend": 6, "fs_annot": 6,
    "label_bold": False, "theta_italic": True,
    "title": "", "title_fs": 7,
    "panel_label": "", "panel_pos": "tl_out", "panel_fs": 8, "panel_bold": True,
    # 標註
    "peaks": "", "peak_rot": "90", "peak_color": "curve", "peak_pad": 0.04, "peak_tol": 0.3,
    "vlines": "", "vline_style": "--", "vline_color": "#8e8e93", "vline_lw": 0.5,
    "spans": "", "span_color": "#d1d1d6", "span_alpha": 0.35,
    "refs": [], "ref_ratio": 0.16, "ref_name_pos": "right", "ref_lw": 0.75,
    "inset": False, "inset_x1": None, "inset_x2": None, "inset_loc": "upper right",
    "inset_w": 0.36, "inset_h": 0.40, "inset_mode": "overlay", "inset_mark": True,
    "texts": "",
    # 輸出
    "preset": "nature1", "width_mm": 89.0, "height_mm": 75.0,
    "margins": "auto", "m_left": 12.0, "m_right": 3.0, "m_top": 3.0, "m_bottom": 10.0,
    "fmt": "png", "dpi": 600, "transparent": False, "fig_name": "XRD_figure",
    "csv_trim": 16, "csv_header": False,
}

# 期刊樣式：寬度、字型、字級、線寬依各期刊作者須知
#   Nature：單欄 89 mm、雙欄 183 mm，Arial / Helvetica 5–7 pt，分圖編號 8 pt 粗體小寫
#   ACS：單欄 3.25 in、雙欄 7 in，Arial / Helvetica，字 ≥ 4.5 pt，線 ≥ 0.5 pt
#   Elsevier：單欄 90 mm、1.5 欄 140 mm、雙欄 190 mm，一般文字 7 pt，線稿 1000 dpi
_NAT = dict(font="Arial", fs_label=7, fs_tick=6, fs_legend=6, fs_annot=6, title_fs=7, panel_fs=8, panel_bold=True,
            lw=0.75, spine_lw=0.5, tick_width=0.5, tick_major_len=3.0, tick_minor_len=1.6, dpi=600)
_ACS = dict(font="Arial", fs_label=8, fs_tick=7, fs_legend=7, fs_annot=7, title_fs=8, panel_fs=9, panel_bold=True,
            lw=0.8, spine_lw=0.75, tick_width=0.75, tick_major_len=3.5, tick_minor_len=2.0, dpi=600)
_ELS = dict(font="Arial", fs_label=7, fs_tick=7, fs_legend=7, fs_annot=7, title_fs=8, panel_fs=8, panel_bold=True,
            lw=0.75, spine_lw=0.6, tick_width=0.6, tick_major_len=3.0, tick_minor_len=1.6, dpi=1000)
PRESETS = {
    "nature1": dict(label="Nature｜單欄 89 mm", width_mm=89, height_mm=75, **_NAT),
    "nature2": dict(label="Nature｜雙欄 183 mm", width_mm=183, height_mm=110, **_NAT),
    "acs1": dict(label="ACS｜單欄 3.25 in", width_mm=82.55, height_mm=70, **_ACS),
    "acs2": dict(label="ACS｜雙欄 7 in", width_mm=177.8, height_mm=105, **_ACS),
    "els1": dict(label="Elsevier｜單欄 90 mm", width_mm=90, height_mm=75, **_ELS),
    "els15": dict(label="Elsevier｜1.5 欄 140 mm", width_mm=140, height_mm=95, **_ELS),
    "els2": dict(label="Elsevier｜雙欄 190 mm", width_mm=190, height_mm=110, **_ELS),
    "slide": dict(label="簡報 / 海報 160 × 100 mm", width_mm=160, height_mm=100, font="Arial", fs_label=14,
                  fs_tick=12, fs_legend=12, fs_annot=12, title_fs=15, panel_fs=16, panel_bold=True, lw=1.5,
                  spine_lw=1.2, tick_width=1.2, tick_major_len=5.0, tick_minor_len=3.0, dpi=300),
}

PALETTES = {
    "okabe": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"],
    "npg": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
            "#7E6148", "#B09C85"],
    "black": ["#000000"],
}

# 期刊指定的無襯線 (Arial / Helvetica) 與襯線 (Times New Roman) 字型。
# 瀏覽器版使用尺寸完全相同的開源字型 Liberation Sans / Serif。
FONT_STACKS = {
    "Arial": ["Arial", "Helvetica", "Liberation Sans", "Arimo", "DejaVu Sans"],
    "Times New Roman": ["Times New Roman", "Times", "Liberation Serif", "Tinos", "DejaVu Serif"],
}
SVG_FONT_ALIAS = {
    "'Liberation Sans'": "Arial, Helvetica, 'Liberation Sans', sans-serif",
    "'Liberation Serif'": "'Times New Roman', Times, 'Liberation Serif', serif",
}

TYPE_LABEL = {"sync_raw": "同步輻射 · 原始", "sync_conv": "同步輻射 · 已轉換", "inhouse": "In-house"}
MARKER_RE = re.compile(r"^\[([v^<>oDdsp*+xh])\]\s*(.*)$")


class UserError(Exception):
    """可直接顯示給使用者的錯誤訊息。"""


# ═════════════════════════════════════════════════════════════════════════════
# 3. 讀檔與格式判斷
# ═════════════════════════════════════════════════════════════════════════════
DATA = {}  # id -> dataset（瀏覽器版：每位使用者各自的記憶體，不會上傳）
SPLIT_RE = re.compile(r"[\s,;]+")
WL_RE = re.compile(
    r"(wave\s*-?\s*length|lambda|λ|k\s*-?\s*alpha\s*1|kα1|wl1|alpha1)[^0-9\n]{0,16}?"
    r"(\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I)


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def wl(v):
    """波長顯示：最多 6 位小數、去掉尾端 0。"""
    return ("%.6f" % float(v)).rstrip("0").rstrip(".")


def read_xrd_text(text):
    """
    讀取任一種文字格式的 XRD 檔，回傳 (標頭行, 數據列[字串欄位])。

    原程式 process_file() 用 lines[3:] 固定刪掉前三行，只適用 .xye（/* … */ 共三行）。
    這裡改成「逐行判斷：前兩欄都是數字才算數據」，對 .xye 的結果完全相同，
    同時能讀 .xyn（兩行標頭）與 In-house .txt（標頭行數不固定、可能以逗號或 Tab 分隔），
    也不會再把空白行輸出成 CSV 的空白列。
    """
    header, rows = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        toks = [t for t in SPLIT_RE.split(s) if t]
        if len(toks) >= 2 and _isnum(toks[0]) and _isnum(toks[1]):
            rows.append(toks)
        elif not rows:
            header.append(s)
    return header, rows


def header_wavelength(htxt):
    """從標頭找波長（Å）；支援 pyFAI 以公尺記錄的寫法，例如 Wavelength: 7.7491e-11。"""
    for m in WL_RE.finditer(htxt):
        try:
            v = float(m.group(2))
        except ValueError:
            continue
        if 0 < v < 1e-6:  # 公尺
            v *= 1e10
        if 0.1 < v < 3.5:
            return v
    return None


def detect_type(name, header):
    """依副檔名與標頭判斷：同步輻射原始 / 同步輻射已轉換 / In-house。"""
    ext = os.path.splitext(name)[1].lower()
    htxt = "\n".join(header)
    hl = htxt.lower()
    hdr_wl = header_wavelength(htxt)

    if ext == ".xyn" or "new_wave_length" in hl or "new wave length" in hl:
        why = []
        if ext == ".xyn":
            why.append("副檔名 .xyn（WinPLOTR 輸出）")
        if "new_wave_length" in hl or "new wave length" in hl:
            why.append("標頭含 new_wave_length")
        return dict(type="sync_conv", hdr_wl=None, reason="、".join(why) + " → 已完成波長修正，不再轉換")

    if ext == ".xye" or hl.startswith("/*"):
        if hdr_wl and abs(hdr_wl - TAG_WL) < 0.002:
            return dict(type="sync_conv", hdr_wl=None,
                        reason="標頭記載 λ = %s Å，已是 Cu Kα → 視為已修正" % wl(hdr_wl))
        r = "副檔名 .xye → 同步輻射原始積分數據，需轉換波長"
        r += "（標頭記載 λ = %s Å）" % wl(hdr_wl) if hdr_wl else "（檔案未記載波長，使用設定值）"
        return dict(type="sync_raw", hdr_wl=hdr_wl, reason=r)

    if ext == ".csv":
        return dict(type="sync_conv", hdr_wl=None,
                    reason=".csv → 視為已換算成 In-house 波長的數據，不再轉換")

    r = "副檔名 %s → In-house 實驗室 XRD，不需轉換" % (ext or "(無)")
    if hdr_wl and abs(hdr_wl - TAG_WL) > 0.01:
        r = "In-house 數據，標頭記載 λ = %s Å（非 Cu Kα），將換算到 Cu Kα1" % wl(hdr_wl)
    else:
        hdr_wl = None  # Cu Kα1 / Kα 平均值皆視為 Cu，不做微小換算
    return dict(type="inhouse", hdr_wl=hdr_wl, reason=r)


def default_label(name, dtype, trim):
    """與原程式相同的命名規則：os.path.basename(filepath)[:-chars_to_remove]。"""
    stem = os.path.splitext(name)[0]
    trim = int(trim or 0)
    if dtype in ("sync_raw", "sync_conv") and 0 < trim < len(name):
        base = name[:-trim]  # 原程式在 chars_to_remove = 0 時會得到空字串，這裡改為保留主檔名
    else:
        base = stem
    return base or stem or "data"


def load_dataset(name, text, trim):
    header, rows = read_xrd_text(text)
    good = [r for r in rows if math.isfinite(float(r[0])) and math.isfinite(float(r[1]))]
    if len(good) < 3:
        raise UserError("%s：找不到兩欄以上的數值數據（2θ, 強度）" % name)
    x = np.array([float(r[0]) for r in good])
    y = np.array([float(r[1]) for r in good])
    det = detect_type(name, header)
    return dict(id=uuid.uuid4().hex[:10], name=name, ext=os.path.splitext(name)[1].lower(),
                header=header, rows=good, x=x, y=y, det=det,
                base=default_label(name, det["type"], trim))


def dataset_meta(d):
    return dict(id=d["id"], name=d["name"], ext=d["ext"], type=d["det"]["type"],
                type_label=TYPE_LABEL[d["det"]["type"]], reason=d["det"]["reason"],
                hdr_wl=d["det"].get("hdr_wl"), n=int(len(d["x"])),
                xr=[float(d["x"].min()), float(d["x"].max())], base=d["base"])


def verify_pairs(new_ids, src_wl):
    """同主檔名的 .xye（原始）與 .xyn（已修正）互相比對，檢查 .xyn 是否真的已轉換、用了什麼波長。"""
    notes = []
    tgt_wl = TAG_WL
    by_stem = {}
    for d in DATA.values():
        by_stem.setdefault(os.path.splitext(d["name"])[0].lower(), []).append(d)
    for ds in by_stem.values():
        raws = [d for d in ds if d["det"]["type"] == "sync_raw"]
        convs = [d for d in ds if d["det"]["type"] == "sync_conv" and d["ext"] == ".xyn"]
        for r in raws:
            for c in convs:
                if r["id"] not in new_ids and c["id"] not in new_ids:
                    continue
                ids = [r["id"], c["id"]]
                if len(r["x"]) != len(c["x"]):
                    notes.append(dict(level="info", ids=ids,
                                      text="%s 與 %s 點數不同，無法逐點比對。" % (r["name"], c["name"])))
                    continue
                src = r["det"].get("hdr_wl") or src_wl
                with np.errstate(invalid="ignore"):
                    xc = convert_two_theta(r["x"], src, tgt_wl)
                dx = float(np.nanmax(np.abs(xc - c["x"])))
                same_y = float(np.max(np.abs(r["y"] - c["y"])) / max(np.max(np.abs(r["y"])), 1e-30)) < 1e-4
                if dx < 2e-4 and same_y:
                    notes.append(dict(level="ok", ids=ids, text=(
                        "已比對 %s ↔ %s：兩者為同一份數據，.xyn 已完成波長修正（2θ 最大差 %.1e°）。建議只顯示其中一個。"
                        % (r["name"], c["name"], dx))))
                    continue
                ok = r["x"] > 1.0
                with np.errstate(invalid="ignore", divide="ignore"):
                    s_r = np.sin(np.radians(r["x"][ok] / 2))
                    s_c = np.sin(np.radians(c["x"][ok] / 2))
                    est_tgt = float(np.nanmedian(src * s_c / s_r))
                    est_src = float(np.nanmedian(tgt_wl * s_r / s_c))
                    res_tgt = float(np.nanmax(np.abs(convert_two_theta(r["x"], src, est_tgt) - c["x"])))
                if res_tgt < 2e-4 and abs(est_tgt - tgt_wl) < 0.01:
                    notes.append(dict(level="ok", ids=ids, text=(
                        "已比對 %s ↔ %s：兩者為同一份數據，.xyn 已完成波長修正。WinPLOTR 以 λ ≈ %.5f Å 換算，"
                        "與本程式的 %s Å 相比 2θ 最大差 %.4f°，小於步距，可忽略。建議只顯示其中一個。"
                        % (r["name"], c["name"], est_tgt, wl(tgt_wl), dx))))
                else:
                    notes.append(dict(level="warn", ids=ids, text=(
                        "已比對 %s ↔ %s：以同步輻射波長 %s Å 換算後與 .xyn 最大差 %.4f°。由 .xyn 反推的原始波長約 "
                        "%.5f Å，請確認該次 beamtime 的波長。%s"
                        % (r["name"], c["name"], wl(src), dx, est_src,
                           "" if same_y else "（強度欄也不同，可能不是同一份數據）"))))
    return notes


# ═════════════════════════════════════════════════════════════════════════════
# 4. 小工具
# ═════════════════════════════════════════════════════════════════════════════
def fnum(v, d=None):
    if v is None or v == "" or isinstance(v, bool):
        return d
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except (TypeError, ValueError):
        return d


def merge_settings(s):
    S = json.loads(json.dumps(DEFAULTS))
    if isinstance(s, dict):
        for k, v in s.items():
            if k in DEFAULTS:
                S[k] = v
    for k, dv in DEFAULTS.items():
        if isinstance(dv, bool):
            S[k] = bool(S[k])
        elif isinstance(dv, (int, float)):
            S[k] = fnum(S[k], dv)
        elif dv is None:
            S[k] = fnum(S[k], None)
        elif isinstance(dv, str):
            S[k] = "" if S[k] is None else str(S[k])
    if not isinstance(S["refs"], list):
        S["refs"] = []
    if not S["src_wl"] or S["src_wl"] <= 0:
        S["src_wl"] = ORI_WL
    if S["font"] not in FONT_STACKS:
        S["font"] = "Arial"
    return S


_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def get_cmap(name):
    try:
        return matplotlib.colormaps[name]
    except Exception:  # matplotlib < 3.5
        from matplotlib import cm
        return cm.get_cmap(name)


def palette_colors(name, n):
    n = max(1, n)
    if name in PALETTES:
        base = PALETTES[name]
        return [base[i % len(base)] for i in range(n)]
    if name == "gray":
        vals = np.linspace(0.0, 0.62, n) if n > 1 else [0.0]
        return [mcolors.to_hex((v, v, v)) for v in vals]
    cmap = get_cmap("viridis")
    vals = np.linspace(0.0, 0.88, n) if n > 1 else [0.0]
    return [mcolors.to_hex(cmap(v)) for v in vals]


def register_fonts(folder):
    """把資料夾內的 .ttf 加入 matplotlib（瀏覽器版用來載入 Liberation 字型）。"""
    n = 0
    if folder and os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".ttf", ".otf")):
                try:
                    font_manager.fontManager.addfont(os.path.join(folder, f))
                    n += 1
                except Exception:
                    pass
    global _FONTS
    _FONTS = None
    return n


_FONTS = None


def available_fonts():
    global _FONTS
    if _FONTS is None:
        _FONTS = {f.name for f in font_manager.fontManager.ttflist}
    return _FONTS


def font_stack(name):
    av = available_fonts()
    stack = FONT_STACKS.get(name, FONT_STACKS["Arial"])
    main = next((f for f in stack if f in av), "DejaVu Sans")
    return main, [f for f in stack if f in av] or ["DejaVu Sans"]


def make_rc(S):
    main, fam = font_stack(S["font"])
    tw = S["tick_width"]
    rc = {
        "font.family": fam, "font.size": S["fs_label"],
        "mathtext.fontset": "custom", "mathtext.rm": main, "mathtext.sf": main,
        "mathtext.it": main + ":italic", "mathtext.bf": main + ":bold", "mathtext.default": "regular",
        "axes.linewidth": S["spine_lw"], "axes.unicode_minus": True, "axes.labelpad": 3.0,
        "axes.edgecolor": "black", "axes.labelcolor": "black", "text.color": "black",
        "lines.linewidth": S["lw"], "lines.solid_joinstyle": "round", "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "legend.fontsize": S["fs_legend"],
    }
    for ax in ("xtick", "ytick"):
        rc.update({
            ax + ".direction": S["tick_dir"], ax + ".major.size": S["tick_major_len"],
            ax + ".minor.size": S["tick_minor_len"], ax + ".major.width": tw,
            ax + ".minor.width": tw * 0.8, ax + ".major.pad": 2.5, ax + ".minor.pad": 2.5,
            ax + ".color": "black", ax + ".labelsize": S["fs_tick"],
        })
    return rc


def fmt_label(s, S):
    s = str(s or "")
    if S.get("_escape_math"):
        return s.replace("$", r"\$")
    if S["theta_italic"] and "$" not in s:
        s = s.replace("θ", r"$\mathit{\theta}$")
    return s


def safe_name(s, default="XRD_figure"):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(s or "")).strip(" .")
    return s or default


# ═════════════════════════════════════════════════════════════════════════════
# 5. 數據處理
# ═════════════════════════════════════════════════════════════════════════════
def effective(d, S):
    """依自動判斷結果決定來源波長與是否需要換算（目標固定為 In-house Cu Kα1）。"""
    t = d["det"]["type"]
    if t == "sync_raw":
        src = d["det"].get("hdr_wl") or S["src_wl"]
    elif t == "sync_conv":
        src = TAG_WL
    else:
        src = d["det"].get("hdr_wl") or TAG_WL
    return {"type": t, "src": float(src), "tgt": TAG_WL, "convert": abs(src - TAG_WL) > 1e-7}


def converted_xy(d, eff):
    x, y = d["x"].astype(float), d["y"].astype(float)
    dropped = 0
    if eff["convert"]:
        with np.errstate(invalid="ignore"):
            x = convert_two_theta(x, eff["src"], eff["tgt"])
        ok = np.isfinite(x)
        dropped = int((~ok).sum())
        x, y = x[ok], y[ok]
    return x, y, dropped


def savgol(y, win, order):
    n = len(y)
    win = int(win)
    order = int(max(0, min(order, 6)))
    if win % 2 == 0:
        win += 1
    win = max(win, order + 3 if (order + 3) % 2 else order + 2)
    if win >= n:
        return y
    k = win // 2
    A = np.vander(np.arange(-k, k + 1, dtype=float), order + 1, increasing=True)
    c = np.linalg.pinv(A)[0]
    return np.convolve(np.pad(y, k, mode="edge"), c[::-1], mode="valid")


def moving_avg(y, win):
    win = int(max(1, win))
    if win % 2 == 0:
        win += 1
    if win <= 1 or win >= len(y):
        return y
    k = win // 2
    return np.convolve(np.pad(y, k, mode="edge"), np.ones(win) / win, mode="valid")


def snip_background(y, m):
    """SNIP 背景估計（Ryan et al. 1988），只用 numpy。"""
    y = np.asarray(y, float)
    off = min(float(y.min()), 0.0)
    v = np.log(np.log(np.sqrt(y - off + 1.0) + 1.0) + 1.0)
    n = len(v)
    for p in range(1, int(m) + 1):
        if 2 * p >= n:
            break
        b = (v[:-2 * p] + v[2 * p:]) / 2.0
        v[p:-p] = np.minimum(v[p:-p], b)
    return (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0 + off


def normalize(x, y, S, x0, x1):
    mode = S["norm"]
    if mode == "none":
        return y
    if mode == "range":
        a = S["norm_x1"] if S["norm_x1"] is not None else x0
        b = S["norm_x2"] if S["norm_x2"] is not None else x1
        m = (x >= min(a, b)) & (x <= max(a, b))
    elif S["norm_scope"] == "view":
        m = (x >= x0) & (x <= x1)
    else:
        m = np.ones_like(x, bool)
    if m.sum() < 2:
        m = np.ones_like(x, bool)
    ys = y[m]
    if mode in ("max", "range", "max100"):
        f = float(ys.max())
        if f <= 0:
            return y
        return y / f * (100.0 if mode == "max100" else 1.0)
    if mode == "minmax":
        lo, hi = float(ys.min()), float(ys.max())
        return (y - lo) / (hi - lo) if hi > lo else y - lo
    if mode == "area":
        A = float(_trapz(ys, x[m]))
        return y / A if A > 0 else y
    return y


def prepare(S, items, warnings):
    """處理所有可見曲線：換算 → 位移 → 平滑 → 背景 → 強度轉換 → 正規化 → 倍率 → 堆疊位移。"""
    pairs = [(it, DATA[it.get("id")]) for it in (items or []) if it.get("id") in DATA]
    pal = palette_colors(S["palette"], len(pairs))
    colors, effs, pre = {}, {}, []
    for i, (it, d) in enumerate(pairs):
        c = it.get("color") or ""
        if not (isinstance(c, str) and mcolors.is_color_like(c)):
            c = pal[i]
        colors[d["id"]] = mcolors.to_hex(c)
        eff = effective(d, S)
        effs[d["id"]] = eff
        if not it.get("visible", True):
            continue
        x, y, dropped = converted_xy(d, eff)
        if dropped:
            lim = 2 * math.degrees(math.asin(min(1.0, eff["src"] / eff["tgt"]))) if eff["tgt"] > eff["src"] else 180
            warnings.append("%s：有 %d 點原始 2θ > %.1f°，換算到 %s Å 後超出可量測範圍（sin θ > 1），已略過。"
                            % (d["name"], dropped, lim, wl(eff["tgt"])))
        if len(x) < 3:
            warnings.append("%s：換算後沒有足夠的數據點。" % d["name"])
            continue
        x = x + (fnum(it.get("shift"), 0.0) or 0.0)
        if np.any(np.diff(x) < 0):
            o = np.argsort(x)
            x, y = x[o], y[o]
        label = it.get("label")
        if label is None:
            label = d["base"]
        pre.append(dict(id=d["id"], idx=i + 1, it=it, d=d, label=str(label), color=colors[d["id"]], x=x, y=y))

    if pre:
        auto = (min(float(p["x"].min()) for p in pre), max(float(p["x"].max()) for p in pre))
    else:
        auto = (0.0, 90.0)
    x0 = S["xmin"] if S["xmin"] is not None else auto[0]
    x1 = S["xmax"] if S["xmax"] is not None else auto[1]
    if x1 <= x0:
        warnings.append("X 軸最小值須小於最大值，已改用自動範圍。")
        x0, x1 = auto

    for p in pre:
        x, y = p["x"], p["y"].copy()
        step = float(np.median(np.diff(x))) if len(x) > 1 else 0.02
        step = step if step > 0 else 0.02
        if S["smooth"] == "sg":
            y = savgol(y, S["smooth_win"], S["smooth_order"])
        elif S["smooth"] == "ma":
            y = moving_avg(y, S["smooth_win"])
        if S["bg"] == "snip":
            m = int(round(max(S["bg_width"], step) / step / 2.0))
            y = y - snip_background(y, max(1, min(m, len(y) // 2 - 1)))
        elif S["bg"] == "min":
            mv = (x >= x0) & (x <= x1)
            y = y - float(y[mv].min() if mv.any() else y.min())
        if S["transform"] == "sqrt":
            y = np.sqrt(np.clip(y, 0, None))
        elif S["transform"] == "log":
            floor = max(float(np.nanmax(y)) * 1e-6, 1e-12)
            y = np.log10(np.clip(y, floor, None))
        y = normalize(x, y, S, x0, x1)
        y = y * fnum(p["it"].get("scale"), 1.0)
        mv = np.where((x >= x0) & (x <= x1))[0]
        if len(mv) < 2:
            warnings.append("%s：在 2θ %.2f–%.2f° 範圍內沒有數據。" % (p["d"]["name"], x0, x1))
            p["xv"], p["yv"] = np.array([]), np.array([])
        else:
            a, b = max(0, mv[0] - 1), min(len(x), mv[-1] + 2)
            p["xv"], p["yv"] = x[a:b], y[a:b]
        p["xa"], p["ya"] = x, y
    pre = [p for p in pre if len(p["xv"])]

    mx = [float(np.nanmax(p["yv"][(p["xv"] >= x0) & (p["xv"] <= x1)])) for p in pre
          if ((p["xv"] >= x0) & (p["xv"] <= x1)).any()]
    ref = max(mx) if mx and max(mx) > 0 else 1.0

    disp = list(pre)  # 由下而上
    if S["mode"] == "stack" and S["first_on_top"]:
        disp = disp[::-1]
    bases = []
    for k, p in enumerate(disp):
        if S["mode"] != "stack" or k == 0:
            bases.append(0.0)
            continue
        if S["stack_method"] == "auto":
            q = disp[k - 1]
            lo, hi = max(q["xv"].min(), p["xv"].min(), x0), min(q["xv"].max(), p["xv"].max(), x1)
            if hi > lo:
                g = np.linspace(lo, hi, 3000)
                need = float(np.max(np.interp(g, q["xv"], q["yv"]) - np.interp(g, p["xv"], p["yv"])))
            else:
                need = float(q["yv"].max() - p["yv"].min())
            bases.append(bases[-1] + need + S["stack_gap"] * ref)
        else:
            bases.append(k * S["stack_spacing"] * ref)
    for p, b in zip(disp, bases):
        p["off"] = b + (fnum(p["it"].get("yoff"), 0.0) or 0.0) * ref
        p["yp"] = p["yv"] + p["off"]
    return dict(curves=disp, pre=pre, view=(x0, x1), auto=auto, ref=ref, colors=colors, effs=effs)


# ═════════════════════════════════════════════════════════════════════════════
# 6. 標註解析
# ═════════════════════════════════════════════════════════════════════════════
def parse_peaks(txt):
    out = []
    for ln in str(txt or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[,\t;]", s)]
        if len(parts) == 1:
            parts = s.split(None, 1)
        x = fnum(parts[0])
        if x is None:
            continue
        text = parts[1] if len(parts) > 1 else ""
        cidx = int(parts[2]) if len(parts) > 2 and re.fullmatch(r"\d+", parts[2] or "") else None
        out.append((x, text, cidx))
    return out


def parse_numbers(txt):
    return [v for v in (fnum(t) for t in re.split(r"[\s,;]+", str(txt or ""))) if v is not None]


def parse_spans(txt):
    out = []
    for ln in str(txt or "").splitlines():
        nums = [fnum(t) for t in re.split(r"[\s,;]+|(?<=\d)-(?=\d)|~|–", ln.strip()) if t]
        nums = [v for v in nums if v is not None]
        if len(nums) >= 2:
            out.append((min(nums[0], nums[1]), max(nums[0], nums[1])))
    return out


def parse_free(txt):
    out = []
    for ln in str(txt or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 3:
            continue
        fx, fy = fnum(parts[0]), fnum(parts[1])
        if fx is None or fy is None:
            continue
        color, text = "black", ",".join(parts[2:])
        if len(parts) >= 4 and parts[-1].startswith("#") and mcolors.is_color_like(parts[-1]):
            color, text = parts[-1], ",".join(parts[2:-1])
        out.append((fx, fy, text.strip(), color))
    return out


def parse_refs(S):
    out = []
    for r in S.get("refs") or []:
        if not isinstance(r, dict) or r.get("hidden"):
            continue
        kind = r.get("kind") or "2theta"
        lam_in = fnum(r.get("wl"), None)
        xs, Is, hk = [], [], []
        for ln in str(r.get("data") or "").splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            toks = [t for t in re.split(r"[\s,;]+", s) if t]
            v = fnum(toks[0])
            if v is None:
                continue
            inten = fnum(toks[1], None) if len(toks) > 1 else None
            hkl = " ".join(toks[2:] if inten is not None else toks[1:])
            inten = 100.0 if inten is None else inten
            if kind == "d":
                arg = TAG_WL / (2.0 * v) if v > 0 else 2
                if arg > 1:
                    continue
                tt = 2.0 * math.degrees(math.asin(arg))
            else:
                tt = v
                if lam_in and abs(lam_in - TAG_WL) > 1e-7:
                    with np.errstate(invalid="ignore"):
                        tt = float(convert_two_theta(v, lam_in, TAG_WL))
                    if not math.isfinite(tt):
                        continue
            xs.append(tt)
            Is.append(inten)
            hk.append(hkl)
        if not xs:
            continue
        Is = np.array(Is, float)
        if Is.max() > 0:
            Is = Is / Is.max() * 100.0
        color = r.get("color") if mcolors.is_color_like(r.get("color") or "") else "#000000"
        out.append(dict(name=str(r.get("name") or ""), color=color, x=np.array(xs), I=Is, hkl=hk,
                        show_hkl=bool(r.get("show_hkl"))))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 7. 繪圖
# ═════════════════════════════════════════════════════════════════════════════
INSET_POS = {
    "upper right": lambda w, h, m: [1 - w - m, 1 - h - m], "upper left": lambda w, h, m: [m, 1 - h - m],
    "upper center": lambda w, h, m: [(1 - w) / 2, 1 - h - m],
    "center right": lambda w, h, m: [1 - w - m, (1 - h) / 2], "center left": lambda w, h, m: [m, (1 - h) / 2],
    "lower right": lambda w, h, m: [1 - w - m, m * 2.5], "lower left": lambda w, h, m: [m, m * 2.5],
}


def local_max(c, x0, tol):
    m = np.abs(c["xv"] - x0) <= tol
    if m.any():
        return float(c["yp"][m].max())
    return float(np.interp(x0, c["xv"], c["yp"]))


def has_peak(c, x0, tol):
    """此曲線在 x0 ± tol 內是否有明顯的峰（高出附近 ±1.5° 中位數 5% 以上的曲線振幅）。"""
    near = np.abs(c["xv"] - x0) <= tol
    wide = np.abs(c["xv"] - x0) <= max(1.5, 4 * tol)
    if not near.any() or wide.sum() < 5:
        return False
    span = float(np.ptp(c["yv"])) or 1.0
    return (float(c["yv"][near].max()) - float(np.median(c["yv"][wide]))) / span > 0.05


def fit_text_ylim(ax, base, texts, marks, renderer, dpi):
    """把框內的文字/標記全部容納進 Y 範圍（只往上延伸）。"""
    y0, y1 = base
    H = ax.bbox.height
    if H <= 0 or (not texts and not marks):
        return
    ax.set_ylim(y0, y1)
    need = y1
    pad_px = 1.5 / 72 * dpi
    for t in texts:
        if not t.get_visible() or not t.get_text():
            continue
        bb = t.get_window_extent(renderer)
        xa, ya = t.get_position()
        anchor = ax.transData.transform((xa, ya))[1]
        f = (max(0.0, bb.y1 - anchor) + pad_px) / H
        if f < 0.85:
            need = max(need, (ya - y0 * f) / (1 - f))
    for (_, ya, ms) in marks:
        f = (ms / 72 * dpi * 1.05 + pad_px) / H
        if f < 0.85:
            need = max(need, (ya - y0 * f) / (1 - f))
    ax.set_ylim(y0, need)


def fit_layout(fig, gs, axes, renderer, pad_in):
    bbs = []
    for a in axes:
        try:
            b = a.get_tightbbox(renderer)
        except Exception:
            b = None
        if b is not None and np.all(np.isfinite(b.bounds)) and b.width > 0 and b.height > 0:
            bbs.append(b)
    if not bbs:
        return False
    bb = Bbox.union(bbs)
    W, H = fig.bbox.width, fig.bbox.height
    p = pad_in * fig.dpi
    dl, dr = (p - bb.x0) / W, (bb.x1 - (W - p)) / W
    db, dt = (p - bb.y0) / H, (bb.y1 - (H - p)) / H
    if max(abs(dl), abs(dr), abs(db), abs(dt)) < 0.3 / max(W, H):
        return False
    l, r = gs.left + dl, gs.right - dr
    b_, t = gs.bottom + db, gs.top - dt
    if r - l < 0.15:
        l, r = 0.3, 0.45
    if t - b_ < 0.15:
        b_, t = 0.3, 0.45
    gs.update(left=l, right=r, bottom=b_, top=t)
    for a in axes:  # 新版 matplotlib 的 gs.update() 不會移動既有座標軸，需手動套用
        ss = a.get_subplotspec()
        if ss is not None:
            a.set_position(ss.get_position(fig))
    return True


def build_figure(S, items, dpi, warnings):
    P = prepare(S, items, warnings)
    curves = P["curves"]
    if not curves:
        raise UserError("沒有可顯示的曲線：請上傳檔案，或確認至少有一個檔案設為顯示。")
    x0, x1 = P["view"]
    ref = P["ref"]
    fs_label, fs_tick, fs_leg, fs_ann = S["fs_label"], S["fs_tick"], S["fs_legend"], S["fs_annot"]
    L = lambda s: fmt_label(s, S)  # noqa: E731
    weight = "bold" if S["label_bold"] else "normal"

    W_in = max(20.0, S["width_mm"]) / 25.4
    H_in = max(15.0, S["height_mm"]) / 25.4
    fig = Figure(figsize=(W_in, H_in), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("white")

    refs = parse_refs(S)
    nref = len(refs)
    manual = S["margins"] == "manual"
    if manual:
        W_mm, H_mm = W_in * 25.4, H_in * 25.4
        box = dict(left=S["m_left"] / W_mm, right=1 - S["m_right"] / W_mm,
                   bottom=S["m_bottom"] / H_mm, top=1 - S["m_top"] / H_mm)
        if box["right"] - box["left"] < 0.1 or box["top"] - box["bottom"] < 0.1:
            warnings.append("固定邊距過大，已改用自動邊距。")
            manual = False
    if not manual:
        box = dict(left=0.14, right=0.96, bottom=0.15, top=0.96)
    gs = GridSpec(1 + nref, 1, figure=fig, height_ratios=[1.0] + [max(0.03, S["ref_ratio"])] * nref,
                  hspace=0.0, **box)
    ax = fig.add_subplot(gs[0])
    raxes = [fig.add_subplot(gs[i + 1], sharex=ax) for i in range(nref)]
    axes = [ax] + raxes
    bottom_ax = axes[-1]
    fitmap = {ax: ([], [])}

    # —— 區間陰影、參考線（在曲線下方）
    spans = parse_spans(S["spans"])
    vlines = parse_numbers(S["vlines"])
    for a in axes:
        for (s0, s1) in spans:
            a.axvspan(s0, s1, color=S["span_color"], alpha=S["span_alpha"], lw=0, zorder=0.5)
        for xv in vlines:
            a.axvline(xv, ls=S["vline_style"], color=S["vline_color"], lw=S["vline_lw"], zorder=1)

    # —— 曲線
    for c in curves:
        if S["fill"]:
            ax.fill_between(c["xv"], c["off"], c["yp"], color=c["color"], alpha=S["fill_alpha"], lw=0, zorder=2)
        ax.plot(c["xv"], c["yp"], color=c["color"], lw=S["lw"], zorder=3)

    ylo = min(float(c["yp"].min()) for c in curves)
    yhi = max(float(c["yp"].max()) for c in curves)
    yr = (yhi - ylo) or 1.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(ylo - S["ymargin_bottom"] * yr, yhi + S["ymargin_top"] * yr)

    # —— 刻度
    xm = S["x_major"]
    if xm and xm > 0 and (x1 - x0) / xm <= 60:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xm))
    else:
        if xm and xm > 0:
            warnings.append("主刻度間距太小（超過 60 個刻度），已改回自動。")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    nm = int(S["x_minor"] or 0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(nm) if nm >= 2 else ticker.NullLocator())
    if S["y_ticks"] and S["frame"] != "bottom":
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2) if nm >= 2 else ticker.NullLocator())
    else:
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())

    frame = S["frame"]
    sec = S["sec_axis"] in ("d", "q")
    for a in axes:
        is_main, is_bottom = a is ax, a is bottom_ax
        for side in ("top", "right", "left", "bottom"):
            a.spines[side].set_linewidth(S["spine_lw"])
        a.spines["top"].set_visible(frame == "box")
        a.spines["right"].set_visible(frame == "box")
        a.spines["left"].set_visible(frame in ("box", "L"))
        ytk = is_main and S["y_ticks"] and frame != "bottom"
        a.tick_params(axis="x", which="both", direction=S["tick_dir"], bottom=True,
                      top=bool(is_main and frame == "box" and S["tick_top"] and not sec),
                      labelbottom=is_bottom, labeltop=False, labelsize=fs_tick)
        a.tick_params(axis="y", which="both", direction=S["tick_dir"], left=bool(ytk),
                      right=bool(ytk and frame == "box" and S["tick_right"]),
                      labelleft=bool(ytk and S["y_ticklabels"]), labelright=False, labelsize=fs_tick)
        if not is_main:
            a.yaxis.set_major_locator(ticker.NullLocator())
            a.yaxis.set_minor_locator(ticker.NullLocator())
    if S["grid"] != "none":
        ax.set_axisbelow(True)
        ax.grid(True, which="major", axis="x", color="#d9d9de", lw=0.4)
        if S["y_ticks"]:
            ax.grid(True, which="major", axis="y", color="#d9d9de", lw=0.4)
        if S["grid"] == "both":
            ax.grid(True, which="minor", axis="x", color="#ececf0", lw=0.3)

    bottom_ax.set_xlabel(L(S["xlabel"]), fontsize=fs_label, fontweight=weight)
    if S["ylabel"]:
        ax.set_ylabel(L(S["ylabel"]), fontsize=fs_label, fontweight=weight)

    # —— 曲線名稱 / 圖例
    lm = S["label_mode"]
    if lm == "auto":
        lm = ("curve" if S["mode"] == "stack" else "legend") if len(curves) > 1 else "none"
    leg_order = list(reversed(curves)) if S["mode"] == "stack" else curves
    xr = x1 - x0
    clabels = []
    if lm in ("curve", "both"):
        for c in curves:
            if not c["label"]:
                continue
            col = c["color"] if S["clabel_color"] else "black"
            pos = S["clabel_pos"]
            if pos == "outside":
                seg = c["yp"][c["xv"] >= x1 - 0.04 * xr]
                yv = float(np.median(seg)) if len(seg) else float(c["yp"][-1])
                ax.text(1.015, yv, L(c["label"]), transform=blended_transform_factory(ax.transAxes, ax.transData),
                        ha="left", va="center", fontsize=fs_leg, color=col, clip_on=False)
            else:
                # 名稱高度 = 曲線靠近該側末端（數據實際存在的範圍）的最高點，避免壓到峰
                if pos == "left":
                    lo = max(x0, float(c["xv"].min()))
                    xs, m, ha = x0 + 0.015 * xr, (c["xv"] >= lo) & (c["xv"] <= lo + 0.14 * xr), "left"
                else:
                    hi = min(x1, float(c["xv"].max()))
                    xs, m, ha = x1 - 0.015 * xr, (c["xv"] <= hi) & (c["xv"] >= hi - 0.14 * xr), "right"
                lvl = float(c["yp"][m].max()) if m.any() else float(np.median(c["yp"]))
                t = ax.text(xs, lvl + S["clabel_pad"] * ref, L(c["label"]), ha=ha, va="bottom",
                            fontsize=fs_leg, color=col, zorder=5)
                fitmap[ax][0].append(t)
                clabels.append((t, c, lvl))
    if lm in ("legend", "both"):
        hl = [(Line2D([], [], color=c["color"], lw=max(S["lw"], 0.8)), L(c["label"])) for c in leg_order
              if c["label"]]
        if hl:
            ncol = max(1, int(S["legend_ncol"] or 1))
            kw = dict(frameon=S["legend_frame"], fancybox=False, edgecolor="black", framealpha=1.0,
                      fontsize=fs_leg, handlelength=S["legend_handle"], handletextpad=0.5,
                      labelspacing=0.35, borderpad=0.45, columnspacing=1.2, borderaxespad=0.6)
            loc = S["legend_loc"]
            if loc == "outside right":
                kw.update(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
            elif loc == "outside top":
                kw.update(loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0,
                          ncol=ncol if ncol > 1 else len(hl))
            else:
                kw["loc"] = loc
            kw.setdefault("ncol", ncol)
            leg = ax.legend([h for h, _ in hl], [t for _, t in hl], **kw)
            if S["legend_frame"]:
                leg.get_frame().set_linewidth(S["spine_lw"])

    # —— 峰位 / 晶面標註
    tol = max(0.01, S["peak_tol"])
    rot = 90 if str(S["peak_rot"]) == "90" else 0
    for (px, text, cidx) in parse_peaks(S["peaks"]):
        if px < x0 or px > x1 or not text:
            continue
        target = next((c for c in curves if c["idx"] == cidx), None) if cidx else None
        if target is None and S["mode"] == "stack":
            # 未指定曲線：由上往下找第一條在此位置真的有峰的曲線
            target = next((c for c in reversed(curves) if has_peak(c, px, tol)), curves[-1])
        elif target is None:
            target = max(curves, key=lambda c: local_max(c, px, tol))
        yy = local_max(target, px, tol) + S["peak_pad"] * ref
        col = target["color"] if S["peak_color"] == "curve" else "black"
        mk = MARKER_RE.match(text)
        if mk:
            ms = fs_ann * 0.62
            ax.plot([px], [yy], marker=mk.group(1), ms=ms, color=col, mec=col, mew=0.6, ls="none", zorder=6,
                    clip_on=False, transform=ax.transData + ScaledTranslation(0, ms / 2 / 72, fig.dpi_scale_trans))
            fitmap[ax][1].append((px, yy, ms))
            if mk.group(2):
                t = ax.text(px, yy, L(mk.group(2)), rotation=rot, ha="center", va="bottom", fontsize=fs_ann,
                            color=col, zorder=6,
                            transform=ax.transData + ScaledTranslation(0, ms * 1.15 / 72, fig.dpi_scale_trans))
                fitmap[ax][0].append(t)
        else:
            t = ax.text(px, yy, L(text), rotation=rot, ha="center", va="bottom", fontsize=fs_ann, color=col,
                        zorder=6)
            fitmap[ax][0].append(t)

    # —— 自由文字
    for (fx, fy, text, color) in parse_free(S["texts"]):
        mk = MARKER_RE.match(text)
        if mk:
            ms = fs_ann * 0.62
            ax.plot([fx], [fy], marker=mk.group(1), ms=ms, color=color, mec=color, ls="none",
                    transform=ax.transAxes, clip_on=False, zorder=7)
            ax.text(fx, fy, L(mk.group(2)), ha="left", va="center", fontsize=fs_ann, color=color, zorder=7,
                    transform=ax.transAxes + ScaledTranslation(ms * 0.9 / 72, 0, fig.dpi_scale_trans))
        else:
            ax.text(fx, fy, L(text), ha="left", va="center", fontsize=fs_ann, color=color,
                    transform=ax.transAxes, zorder=7)

    # —— 上方次座標軸（以 Cu Kα1 計算）
    if sec:
        lam = TAG_WL

        def tt2d(tt):
            s = np.sin(np.radians(np.asarray(tt, float) / 2.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(s > 1e-6, lam / (2.0 * np.where(s > 1e-6, s, 1.0)), 1e6)

        def d2tt(d):
            d = np.asarray(d, float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.clip(lam / (2.0 * np.where(np.abs(d) > 1e-9, d, 1e-9)), -1, 1)
            return 2.0 * np.degrees(np.arcsin(r))

        def tt2q(tt):
            return 4 * np.pi * np.sin(np.radians(np.asarray(tt, float) / 2.0)) / lam

        def q2tt(q):
            return 2.0 * np.degrees(np.arcsin(np.clip(np.asarray(q, float) * lam / (4 * np.pi), -1, 1)))

        funcs, lab = ((tt2d, d2tt), "d (Å)") if S["sec_axis"] == "d" else ((tt2q, q2tt), "q (Å$^{-1}$)")
        sax = ax.secondary_xaxis("top", functions=funcs)
        if S["sec_axis"] == "d":
            # d 與 2θ 非線性：從「好讀」的 d 值中挑出在 2θ 上大致等距的刻度，避免低角度擠在一起
            cands = [50, 30, 20, 15, 10, 8, 6, 5, 4, 3.5, 3, 2.5, 2.2, 2, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1,
                     1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
            pos = sorted(((float(d2tt(v)), v) for v in cands), key=lambda p: p[0])
            chosen, last = [], -1e9
            for tt, v in pos:
                if x0 <= tt <= x1 and tt - last >= (x1 - x0) / 9.0:
                    chosen.append(v)
                    last = tt
            sax.xaxis.set_major_locator(ticker.FixedLocator(chosen))
            sax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: "%g" % v))
            sax.xaxis.set_minor_locator(ticker.NullLocator())
        else:
            sax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            if nm >= 2:
                sax.xaxis.set_minor_locator(ticker.AutoMinorLocator(nm))
        sax.tick_params(which="both", direction=S["tick_dir"], labelsize=fs_tick, width=S["tick_width"])
        sax.tick_params(which="major", length=S["tick_major_len"])
        sax.tick_params(which="minor", length=S["tick_minor_len"], width=S["tick_width"] * 0.8)
        for sp in sax.spines.values():
            sp.set_linewidth(S["spine_lw"])
        sax.set_xlabel(L(lab), fontsize=fs_label, fontweight=weight)

    # —— 放大插圖
    if S["inset"]:
        a, b = S["inset_x1"], S["inset_x2"]
        if a is None or b is None or a == b:
            warnings.append("放大插圖：請設定 2θ 起點與終點。")
        else:
            a, b = min(a, b), max(a, b)
            w = min(0.9, max(0.1, S["inset_w"]))
            h = min(0.9, max(0.1, S["inset_h"]))
            fn = INSET_POS.get(S["inset_loc"], INSET_POS["upper right"])
            axi = ax.inset_axes(fn(w, h, 0.025) + [w, h])
            mode = S["inset_mode"]
            src = curves if mode in ("stack", "same") else P["pre"]
            k = 0
            for c in src:
                m = (c["xa"] >= a) & (c["xa"] <= b)
                if m.sum() < 2:
                    continue
                xi, yi = c["xa"][m], c["ya"][m]
                if mode in ("overlay", "stack"):
                    # 區間內沒有明顯的峰（只有雜訊）時，區間正規化會把雜訊放大成滿版 → 不畫
                    noise = 1.4826 * float(np.median(np.abs(np.diff(yi)))) / math.sqrt(2) if len(yi) > 3 else 0
                    if noise > 0 and float(yi.max() - np.median(yi)) < 6 * noise:
                        warnings.append("放大插圖：%s 在 %.2f–%.2f° 內沒有明顯的峰，未畫入插圖。" % (c["label"], a, b))
                        continue
                    lo, hi = float(yi.min()), float(yi.max())
                    yi = (yi - lo) / (hi - lo) if hi > lo else yi - lo
                    if mode == "stack":
                        yi = yi + k * 1.1
                else:
                    yi = yi + c["off"]
                axi.plot(xi, yi, color=c["color"], lw=S["lw"])
                k += 1
            for xv in vlines:
                if a <= xv <= b:
                    axi.axvline(xv, ls=S["vline_style"], color=S["vline_color"], lw=S["vline_lw"])
            axi.set_xlim(a, b)
            axi.margins(y=0.08)
            axi.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            axi.xaxis.set_minor_locator(ticker.AutoMinorLocator(2) if nm >= 2 else ticker.NullLocator())
            axi.yaxis.set_major_locator(ticker.NullLocator())
            axi.tick_params(which="both", direction=S["tick_dir"], labelsize=fs_tick * 0.85, top=False,
                            right=False, pad=1.5)
            for sp in axi.spines.values():
                sp.set_linewidth(S["spine_lw"])
            if S["inset_mark"]:
                if mode == "same":
                    try:
                        ax.indicate_inset_zoom(axi, edgecolor="#555555", lw=S["spine_lw"] * 0.8)
                    except Exception:
                        ax.axvspan(a, b, color="#000000", alpha=0.06, lw=0, zorder=0.6)
                else:
                    ax.axvspan(a, b, color="#000000", alpha=0.06, lw=0, zorder=0.6)

    # —— 標準卡片（參考圖譜）
    for r, ra in zip(refs, raxes):
        m = (r["x"] >= x0) & (r["x"] <= x1)
        ra.vlines(r["x"][m], 0, r["I"][m], color=r["color"], lw=S["ref_lw"])
        ra.set_ylim(0, 112)
        texts = []
        if r["name"]:
            left = S["ref_name_pos"] == "left"
            ra.text(0.012 if left else 0.988, 0.9, L(r["name"]), transform=ra.transAxes,
                    ha="left" if left else "right", va="top", fontsize=fs_leg, color=r["color"])
        if r["show_hkl"]:
            for xx, ii, hh in zip(r["x"][m], r["I"][m], np.array(r["hkl"], dtype=object)[m]):
                if hh:
                    texts.append(ra.text(xx, ii + 3, L(hh), rotation=90, ha="center", va="bottom",
                                         fontsize=fs_ann * 0.85, color=r["color"]))
        fitmap[ra] = (texts, [])

    # —— 標題 / 分圖編號
    if S["title"]:
        ax.set_title(L(S["title"]), fontsize=S["title_fs"], pad=4)
    panel = None
    if S["panel_label"]:
        pw = "bold" if S["panel_bold"] else "normal"
        if S["panel_pos"] == "tl_in":
            panel = ax.text(0.025, 0.975, S["panel_label"], transform=ax.transAxes, ha="left", va="top",
                            fontsize=S["panel_fs"], fontweight=pw, zorder=8)
        elif S["panel_pos"] == "tr_in":
            panel = ax.text(0.975, 0.975, S["panel_label"], transform=ax.transAxes, ha="right", va="top",
                            fontsize=S["panel_fs"], fontweight=pw, zorder=8)
        else:
            panel = ax.text(0, 1, S["panel_label"], ha="left", va="bottom", fontsize=S["panel_fs"], fontweight=pw,
                            transform=ax.transAxes + ScaledTranslation(0, 2.0 / 72, fig.dpi_scale_trans),
                            clip_on=False)

    # —— 版面：曲線名稱定位 + 文字容納 + 自動邊距
    renderer = fig.canvas.get_renderer()
    base = {a: a.get_ylim() for a in fitmap}
    for it in range(5):
        inv = ax.transData.inverted()
        for (t, c, lvl0) in clabels:  # 依文字實際寬度，放在它所覆蓋那段曲線的最高點之上
            bb = t.get_window_extent(renderer)
            xa, xb = inv.transform((bb.x0, 0))[0], inv.transform((bb.x1, 0))[0]
            m = (c["xv"] >= min(xa, xb)) & (c["xv"] <= max(xa, xb))
            t.set_y((float(c["yp"][m].max()) if m.any() else lvl0) + S["clabel_pad"] * ref)
        for a, (texts, marks) in fitmap.items():
            fit_text_ylim(a, base[a], texts, marks, renderer, fig.dpi)
        if panel is not None and S["panel_pos"] == "tl_out":
            yb = ax.yaxis.get_tightbbox(renderer)
            left_px = min(yb.x0, ax.bbox.x0) if (yb is not None and yb.width > 0) else ax.bbox.x0
            top_px = ax.bbox.y1
            for extra in ([ax.title] if S["title"] else []) + ([sax] if sec else []):
                try:
                    eb = extra.get_tightbbox(renderer)
                    if eb is not None and eb.height > 0:
                        top_px = max(top_px, eb.y1)
                except Exception:
                    pass
            panel.set_position(((left_px - ax.bbox.x0) / ax.bbox.width, (top_px - ax.bbox.y0) / ax.bbox.height))
        if manual:
            if it >= 1:
                break
            continue
        moved = fit_layout(fig, gs, axes, renderer, pad_in=1.2 / 25.4)
        if not moved and it > 0:
            break
    info = dict(colors=P["colors"],
                eff={k: dict(v, type_label=TYPE_LABEL[v["type"]]) for k, v in P["effs"].items()},
                auto=list(P["auto"]), view=list(P["view"]))
    return fig, info, P


def render_bytes(S_in, items, dpi, fmt="png", transparent=False):
    """建立圖並輸出為指定格式的 bytes。若 $…$ 數學式解析失敗，自動改為一般文字重試。"""
    S = merge_settings(S_in)
    for attempt in (0, 1):
        warnings = []
        if attempt:
            S["_escape_math"] = True
            warnings.append("文字中的 $ 無法解析為數學式，已改以一般文字顯示。")
        try:
            with matplotlib.rc_context(make_rc(S)), _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*PostScript backend does not support transparency.*")
                fig, info, P = build_figure(S, items, dpi, warnings)
                buf = io.BytesIO()
                kw = dict(format=fmt, dpi=dpi, transparent=transparent, facecolor="none" if transparent else "white")
                if fmt == "tiff":
                    try:
                        fig.savefig(buf, pil_kwargs={"compression": "tiff_lzw"}, **kw)
                    except Exception:
                        buf = io.BytesIO()
                        fig.savefig(buf, **kw)
                else:
                    fig.savefig(buf, **kw)
                size = (int(round(fig.get_figwidth() * dpi)), int(round(fig.get_figheight() * dpi)))
            data = buf.getvalue()
            if fmt == "svg":  # 讓 Illustrator / Inkscape 開啟時優先使用 Arial / Times New Roman
                txt = data.decode("utf-8")
                for k, v in SVG_FONT_ALIAS.items():
                    txt = txt.replace(k, v)
                data = txt.encode("utf-8")
            return data, info, warnings, size, P
        except UserError:
            raise
        except ValueError as e:
            if attempt == 0 and ("$" in json.dumps(S, ensure_ascii=False) or "mathtext" in str(e).lower()
                                 or "Expected" in str(e)):
                continue
            raise


# ═════════════════════════════════════════════════════════════════════════════
# 8. 找峰、CSV、匯出
# ═════════════════════════════════════════════════════════════════════════════
def find_peaks_simple(x, y, n=10, min_prom=0.04):
    if len(x) < 7:
        return []
    step = float(np.median(np.diff(x))) or 0.02
    w = max(2, int(round(0.12 / step)))
    W = max(w * 4, int(round(1.5 / step)))
    rng = float(y.max() - y.min()) or 1.0
    cand = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            a, b = max(0, i - w), min(len(y), i + w + 1)
            if y[i] < y[a:b].max():
                continue
            prom = (y[i] - max(y[max(0, i - W):i + 1].min(), y[i:min(len(y), i + W + 1)].min())) / rng
            if prom >= min_prom:
                cand.append((prom, float(x[i])))
    cand.sort(reverse=True)
    return sorted(px for _, px in cand[:max(1, int(n))])


def converted_csv_text(d, eff, header=False):
    """
    與原程式 process_file() 相同的輸出：
      第一欄 = 換算後 2θ（f"{…:.6f}"），第二欄 = 原始強度字串，刪除第三欄 (esd)，逗號分隔、無標頭。
    修正：
      * 原程式只在「欄數 > 2」時才換算，若某列只有兩欄，2θ 會沒換算就寫出 → 這裡每列都依判斷結果處理。
      * 原始 2θ 太大時 arcsin 會得到 nan（λ 0.77491 → 1.540598 時，原始 2θ > 60.4° 就無解）→ 略過該列。
    """
    lines = ["2theta,intensity"] if header else []
    for toks in d["rows"]:
        cols = list(toks[:2])
        if eff["convert"]:
            with np.errstate(invalid="ignore"):
                nx = convert_two_theta(float(toks[0]), eff["src"], eff["tgt"])
            if not np.isfinite(nx):
                continue
            cols[0] = f"{nx:.6f}"
        lines.append(",".join(cols))
    return "\n".join(lines) + "\n"


def all_converted_csv(S, items):
    out, used = [], {}
    for it in items or []:
        d = DATA.get(it.get("id"))
        if not d:
            continue
        eff = effective(d, S)
        name = safe_name(default_label(d["name"], eff["type"], S["csv_trim"]), "data") + ".csv"
        if name.lower() in used and d["ext"]:  # 例如 .xye 與 .xyn 同名 → DT…_001_xyn.csv
            name = name[:-4] + "_" + d["ext"].lstrip(".") + ".csv"
        while name.lower() in used:
            used[name.lower()] += 1
            name = name[:-4] + "_%d.csv" % used[name.lower()]
        used.setdefault(name.lower(), 1)
        out.append((name, converted_csv_text(d, eff, S["csv_header"])))
    return out


FMT_EXT = {"png": "png", "tiff": "tif", "pdf": "pdf", "svg": "svg", "eps": "eps"}
FMT_MIME = {"png": "image/png", "tiff": "image/tiff", "pdf": "application/pdf", "svg": "image/svg+xml",
            "eps": "application/postscript"}


def export_figure(S_in, items):
    S = merge_settings(S_in)
    fmt = S["fmt"] if S["fmt"] in FMT_EXT else "png"
    dpi = int(min(2400, max(72, S["dpi"])))
    data, info, warnings, size, P = render_bytes(S_in, items, dpi, fmt, S["transparent"])
    return safe_name(S["fig_name"]) + "." + FMT_EXT[fmt], data, FMT_MIME[fmt]


def make_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
# 9. 礦物鑑定：最強峰 → webmineral 反查 → RRUFF 驗證 → 扣除已確認礦物的峰後再查
# ═════════════════════════════════════════════════════════════════════════════
ELEMENTS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os
Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr""".split())
MINDB = None
MINDB_PATH = None  # 瀏覽器版由 worker 指定；本機版預設為 data/minerals.json.gz
POLYTYPE_RE = re.compile(r"(-\d*[A-Z]{1,2}\d*[a-z]?|-\d+[A-Za-z]*)+$")
ID_TOL = 0.12     # 峰位容許誤差（Cu Kα1 2θ，度）
ID_SHIFT = 0.25   # 允許的整體位移（樣品高度 / 零點），度


def base_name(n):
    return POLYTYPE_RE.sub("", str(n).strip()).lower()


def looks_synthetic(name):
    """AMCSD 也收錄合成化合物（名稱多為化學式，如 Al6Ca9O18、Br Cu、MnSe、Tricalcium aluminate）。"""
    b = POLYTYPE_RE.sub("", str(name).strip())
    if re.search(r"\d", b) or re.search(r"[a-z]\s+[a-z]", b):
        return True
    if b.lower() in ("ice",):
        return False
    toks = re.findall(r"[A-Z][a-z]?|\(|\)|\s+", b)
    return "".join(toks) == b and sum(1 for t in toks if t.strip() and t not in "()") >= 2 and \
        all(t in ELEMENTS for t in toks if t.strip() and t not in "()")


def mindb():
    """載入 RRUFF / AMCSD 精簡峰表（tools/build_mineral_db.py 產生）。"""
    global MINDB
    if MINDB is not None:
        return MINDB
    import gzip
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [MINDB_PATH] if MINDB_PATH else [os.path.join(here, "minerals.json.gz"),
                                              os.path.join(here, "data", "minerals.json.gz")]
    path = next((p for p in paths if p and os.path.exists(p)), None)
    if not path:
        raise UserError("找不到礦物資料庫 minerals.json.gz：請把它放在和 xrd_core.py 同一層"
                        "（或執行 tools/build_mineral_db.py 重新建立）。")
    with open(path, "rb") as f:
        blob = f.read()
    # 有些網頁伺服器會自動解壓 .gz：檢查檔頭，壓縮或未壓縮都能讀
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    try:
        raw = json.loads(blob.decode("utf-8"))
    except ValueError:
        raise UserError("礦物資料庫 minerals.json.gz 已損毀或不完整，請重新下載或重新上傳這個檔案。")
    ents, by_name, by_base = [], {}, {}
    for row in raw["e"]:
        name, src, rid, note, els, pk = row[:6]
        n_r = row[6] if len(row) > 6 else 0
        n_a = row[7] if len(row) > 7 else 0
        rows = [tok.split(":", 2) for tok in pk.split(";") if tok]
        d = np.array([float(r[0]) for r in rows])
        I = np.array([float(r[1]) for r in rows])
        s = TAG_WL / (2.0 * d)
        ok = s < 1
        t = np.full(len(d), np.nan)
        t[ok] = 2.0 * np.degrees(np.arcsin(s[ok]))
        top = np.argsort(-I)[:3]
        e = dict(name=name, src=src, rid=rid, note=note, el=set(els.split()), d=d, I=I, t=t,
                 hkl=[r[2] if len(r) > 2 else "" for r in rows], top3=d[top], key=src + ":" + rid, nrec=n_r + n_a, n_r=n_r, n_a=n_a)
        by_name.setdefault(name.lower(), []).append(len(ents))
        by_base.setdefault(base_name(name), []).append(len(ents))
        ents.append(e)
    MINDB = dict(e=ents, by_name=by_name, by_base=by_base, by_key={e["key"]: e for e in ents},
                 version=raw.get("version", ""), cite=raw.get("cite", []),
                 names=sorted({e["name"] for e in ents}, key=str.lower))
    return MINDB


def lookup(name):
    db = mindb()
    k = str(name).strip().lower()
    idx = db["by_name"].get(k) or db["by_base"].get(base_name(name)) or []
    return [db["e"][i] for i in idx]


def _sliding_max(a, w):
    pad = np.pad(a, w, mode="edge")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(pad, 2 * w + 1).max(axis=1)
    except ImportError:  # numpy < 1.20
        return np.array([pad[i:i + 2 * w + 1].max() for i in range(len(a))])


def detect_peaks(x, y):
    """扣除 SNIP 背景後找局部最大值：高於 6 倍雜訊、且 ≥ 最強峰 0.4%；以拋物線內插精修峰位。"""
    step = float(np.median(np.diff(x))) or 0.02
    yn = y - snip_background(y, max(1, min(int(round(1.5 / step / 2)), len(y) // 2 - 1)))
    noise = 1.4826 * float(np.median(np.abs(np.diff(yn)))) / math.sqrt(2) or 1e-12
    w = max(2, int(round(0.08 / step)))
    win = _sliding_max(yn, w)
    top = float(yn.max())
    cand = np.where((yn >= win) & (yn > 6 * noise) & (yn > 0.004 * top))[0]
    keep, last = [], -10 ** 9
    for i in cand:
        if i - last > w:
            keep.append(i)
            last = i
    t, h = [], []
    for i in keep:
        off = 0.0
        if 0 < i < len(yn) - 1:
            a, b, c = yn[i - 1], yn[i], yn[i + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den != 0 else 0.0
            off = max(-1.0, min(1.0, off))
        t.append(x[i] + off * step)
        h.append(float(yn[i]))
    t, h = np.array(t), np.array(h)
    I = 100.0 * h / (h.max() if len(h) else 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = TAG_WL / (2.0 * np.sin(np.radians(t / 2.0)))
    return dict(t=t, d=d, I=I, range=(float(x.min()), float(x.max())), noise=noise)


def observed(S, items, iid):
    if iid not in DATA:
        raise UserError("請先選擇要鑑定的曲線。")
    it = next((i for i in items if i.get("id") == iid), {})
    d0 = DATA[iid]
    x, y, _ = converted_xy(d0, effective(d0, S))
    x = x + (fnum(it.get("shift"), 0.0) or 0.0)
    o = np.argsort(x)
    return detect_peaks(x[o], y[o])


def score_entry(e, obs, tol=ID_TOL, max_shift=ID_SHIFT, detail=False, weights=None):
    """
    把參考圖譜和偵測到的峰比對（Cu Kα1 2θ）：
      峰強覆蓋 = 參考圖譜中有對到觀測峰的線，佔其總強度的比例
      強峰     = 參考圖譜中強度 ≥ 最強 50% 的線，必須全部出現
      偶然機率 = 每根線剛好落在某根觀測峰附近的機率；對上的線越多，越不可能是巧合
    允許整體位移 ±max_shift（樣品高度、零點誤差）。
    """
    t_obs, I_obs = obs["t"], obs["I"]
    lo, hi = obs["range"]
    if len(t_obs) < 2:
        return None
    t, I = e["t"], e["I"]
    m = np.isfinite(t) & (t >= lo + 0.2) & (t <= hi - 0.2) & (I >= 3)
    if m.sum() < 2 or I[m].max() < 20:
        return dict(testable=False, score=0.0, verdict="na",
                    reason="此礦物的主要繞射峰不在量測範圍 %.1f–%.1f° 內" % (lo, hi))
    idx = np.where(m)[0]
    tr, Ir = t[idx], I[idx]
    w_obs = np.sqrt(np.clip(I_obs, 0, None)) if weights is None else weights
    p_rand = min(0.9, len(t_obs) * 2 * tol / max(1.0, hi - lo))
    strong = Ir >= 0.5 * Ir.max()
    best = None
    for s in np.arange(-max_shift, max_shift + 1e-9, 0.02):
        pos = tr + s
        j = np.clip(np.searchsorted(t_obs, pos), 1, len(t_obs) - 1)
        jj = np.where(np.abs(t_obs[j - 1] - pos) <= np.abs(t_obs[j] - pos), j - 1, j)
        dt = np.abs(t_obs[jj] - pos)
        ok = dt <= tol
        cov = float((Ir * ok).sum() / Ir.sum())
        cov_s = float(ok[strong].mean())
        err = float(dt[ok].mean() / tol) if ok.any() else 1.0
        n_sig = int((ok & (Ir >= 5)).sum())
        sig = 1.0 - p_rand ** n_sig
        hit = np.unique(jj[ok])
        expl = float(w_obs[hit].sum() / w_obs.sum()) if w_obs.sum() else 0.0
        sc = cov * cov_s * (1 - 0.4 * err) * sig * (0.75 + 0.25 * min(1.0, expl * 5))
        if best is None or sc > best["score"] + 1e-12:
            best = dict(score=sc, shift=round(float(s), 2), cov=cov, strong=[int(ok[strong].sum()), int(strong.sum())],
                        lines=[int(ok.sum()), int(len(ok))], err=err, hit=hit.tolist(), _ok=ok, _jj=jj, _dt=dt)
    sc = best["score"]
    best.update(testable=True, verdict="high" if sc >= 0.75 else "mid" if sc >= 0.5 else "low")
    if detail:
        order = np.argsort(-Ir)[:14]
        best["table"] = [dict(hkl=e["hkl"][idx[k]], d=round(float(e["d"][idx[k]]), 4), I=round(float(Ir[k])),
                              t=round(float(tr[k] + best["shift"]), 3),
                              obs=round(float(t_obs[best["_jj"][k]]), 3) if best["_ok"][k] else None,
                              diff=round(float(t_obs[best["_jj"][k]] - tr[k] - best["shift"]), 3) if best["_ok"][k] else None)
                         for k in sorted(order, key=lambda k: tr[k])]
    for k in ("_ok", "_jj", "_dt"):
        best.pop(k, None)
    best["score"] = round(sc, 3)
    best["cov"] = round(best["cov"], 3)
    best["err"] = round(best["err"], 3)
    return best


def entry_info(e, sc=None):
    out = dict(key=e["key"], name=e["name"], src="RRUFF" if e["src"] == "R" else "AMCSD", rid=e["rid"],
               note=e["note"], el=sorted(e["el"]))
    if sc:
        out.update({k: v for k, v in sc.items() if k != "hit"})
    return out


def best_entry(entries, obs, tol, weights=None):
    scored = [(score_entry(e, obs, tol, weights=weights), e) for e in entries]
    scored = [(s, e) for s, e in scored if s]
    if not scored:
        return None, []
    scored.sort(key=lambda z: -z[0]["score"])
    return scored[0], scored


def formula_elements(formula):
    return {s for s in re.findall(r"[A-Z][a-z]?", formula or "") if s in ELEMENTS}


def element_ok(els, allowed, required):
    if required and not required <= els:
        return False
    if allowed and not els <= (allowed | {"O", "H"}):
        return False
    return True


def parse_webmineral(pages):
    """解析 webmineral X-Ray 查詢結果頁（D1–D3、I1–I3、礦物名稱、化學式）。"""
    import html as _html
    total, rows = 0, []
    for t in pages:
        m = re.search(r"Listing of\s+(\d+)\s+Records", t)
        if m:
            total = max(total, int(m.group(1)))
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S | re.I):
            tds = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                   for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S | re.I)]
            if len(tds) >= 8 and re.match(r"^\d+\.\d+", tds[0]):
                num = lambda s: fnum((re.match(r"\s*([\d.]+)", s) or [None, None])[1])  # noqa: E731
                link = re.search(r'href="(?:\.\./)?(data/[^"]+)"', tr)
                rows.append(dict(d=[num(tds[0]), num(tds[2]), num(tds[4])], I=[num(tds[1]), num(tds[3]), num(tds[5])],
                                 name=tds[6], formula=re.sub(r"\s+", " ", tds[7]),
                                 url=("https://webmineral.com/" + link.group(1)) if link else ""))
    return total, rows


def explained_by(obs, confirmed, tol):
    """已確認的礦物（含其位移）可解釋哪些觀測峰。"""
    owner = [None] * len(obs["t"])
    for c in confirmed or []:
        e = mindb()["by_key"].get(c.get("key"))
        if not e:
            continue
        eps = fnum(c.get("eps"), 0.0) or 0.0
        m = np.isfinite(e["t"]) & (e["I"] >= 3)
        if eps:
            sn = TAG_WL / (2 * e["d"][m] * (1 + eps))
            tr = 2 * np.degrees(np.arcsin(np.clip(sn, 0, 1))) + fnum(c.get("shift"), 0.0)
        else:
            tr = e["t"][m] + fnum(c.get("shift"), 0.0)
        if not len(tr):
            continue
        for i, tt in enumerate(obs["t"]):
            if owner[i] is None and np.min(np.abs(tr - tt)) <= tol:
                owner[i] = e["name"]
    return owner


def residual_weights(obs, owner):
    """已被確認礦物解釋的峰，權重降到 5%：排序時優先考慮能解釋剩餘峰的候選。"""
    w = np.sqrt(np.clip(obs["I"], 0, None))
    return np.where(np.array([o is not None for o in owner], bool), w * 0.05, w) if len(w) else w


def id_peaks(S, items, a):
    obs = observed(S, items, a.get("id"))
    owner = explained_by(obs, a.get("confirmed"), fnum(a.get("tol"), ID_TOL))
    peaks = [dict(i=i, t=round(float(obs["t"][i]), 3), d=round(float(obs["d"][i]), 4), I=round(float(obs["I"][i]), 1),
                  by=owner[i]) for i in range(len(obs["t"]))]
    return dict(peaks=peaks, range=obs["range"])


def id_search(S, items, a):
    """候選：webmineral 回傳的頁面（a['pages']）或本機資料庫的三強峰篩選；再用 RRUFF / AMCSD 全圖譜驗證排序。"""
    obs = observed(S, items, a.get("id"))
    tol = fnum(a.get("tol"), ID_TOL)
    weights = residual_weights(obs, explained_by(obs, a.get("confirmed"), tol))
    pct = fnum(a.get("pct"), 1.5) / 100.0
    any_order = a.get("any_order", True)
    Ds = [fnum(v) for v in (a.get("ds") or []) if fnum(v)]
    allowed = {s for s in re.split(r"[\s,;]+", a.get("allowed") or "") if s in ELEMENTS}
    required = {s for s in re.split(r"[\s,;]+", a.get("required") or "") if s in ELEMENTS}
    cands, total, source = [], 0, "webmineral"
    if a.get("pages"):
        total, rows = parse_webmineral(a["pages"])
        seen = set()
        for r in rows:
            k = r["name"].lower()
            if k in seen:
                continue
            seen.add(k)
            if not element_ok(formula_elements(r["formula"]), allowed, required):
                continue
            cands.append(dict(name=r["name"], formula=r["formula"], wm_d=r["d"], wm_I=r["I"], url=r["url"]))
        n_rows = len(seen)
    else:
        source = "rruff"
        if not Ds:
            raise UserError("請先在峰位表勾選 1–3 根峰。")
        groups = {}
        for e in mindb()["e"]:
            if len(e["top3"]) < len(Ds) or (e["src"] == "A" and looks_synthetic(e["name"])):
                continue
            orders = itertools.permutations(range(len(e["top3"])), len(Ds)) if any_order else [range(len(Ds))]
            if any(all(abs(e["top3"][j] - Ds[k]) <= pct * Ds[k] for k, j in enumerate(o)) for o in orders):
                groups.setdefault(e["name"].lower(), []).append(e)
        n_rows = total = len(groups)
        for k, es in groups.items():
            if not element_ok(set().union(*(e["el"] for e in es)), allowed, required):
                continue
            cands.append(dict(name=es[0]["name"], formula=" ".join(sorted(es[0]["el"])),
                              wm_d=[round(float(v), 3) for v in es[0]["top3"]], wm_I=None, url=""))
    out = []
    for c in cands:
        entries = lookup(c["name"])
        if not entries:
            out.append(dict(c, in_db=False, best=None, n_entries=0))
            continue
        b, allsc = best_entry(entries, obs, tol, weights)
        out.append(dict(c, in_db=True, n_entries=len(entries),
                        best=entry_info(b[1], b[0]) if b else None))
    out.sort(key=lambda c: (not c["in_db"], -(c["best"]["score"] if c["best"] else -1)))
    return dict(source=source, total=total, found=n_rows, kept=len(out), candidates=out[:200])


def id_verify(S, items, a):
    obs = observed(S, items, a.get("id"))
    tol = fnum(a.get("tol"), ID_TOL)
    if a.get("key"):
        e = mindb()["by_key"].get(a["key"])
        entries = [e] if e else []
    else:
        entries = lookup(a.get("name") or "")
    if not entries:
        raise UserError("RRUFF / AMCSD 資料庫中找不到「%s」。" % (a.get("name") or a.get("key") or ""))
    weights = residual_weights(obs, explained_by(obs, a.get("confirmed"), tol))
    res = []
    for e in entries:
        sc = score_entry(e, obs, tol, detail=True, weights=weights)
        if sc:
            res.append(entry_info(e, sc))
    res.sort(key=lambda r: -r["score"])
    return dict(name=entries[0]["name"], entries=res)


def id_ref(a):
    e = mindb()["by_key"].get(a.get("key"))
    if not e:
        raise UserError("找不到參考資料。")
    lines = ["%.4f %d %s" % (d, I, h) for d, I, h in zip(e["d"], e["I"], e["hkl"])]
    return dict(name=e["name"], src="RRUFF" if e["src"] == "R" else "AMCSD", rid=e["rid"], note=e["note"],
                data="\n".join(lines))


# ═════════════════════════════════════════════════════════════════════════════
# 10. 自動全譜辨識（whole-pattern search-match）
#   1. 全資料庫峰表篩選：每一筆參考資料都比對，容許 2θ 位移與 ±1.2% 晶格縮放（固溶體）
#   2. 每個候選以峰表（含 Kα2）卷積 pseudo-Voigt 產生全譜（q 空間、多種峰寬）
#   3. 逐步加入最能降低擬合殘差的礦物（非負加權最小平方 + 平滑背景），並精修位移、晶格、峰寬
#   4. 依證據分級（確定 / 可能）、反向剔除、列出圖譜幾乎相同的替代礦物
#   常見度：同名礦物在 RRUFF + AMCSD 的資料筆數，罕見礦物需要更強的證據。
# ═════════════════════════════════════════════════════════════════════════════
L1, L2 = TAG_WL, 1.544426  # Cu Kα1, Kα2


def fp_to_q(t): return 4 * np.pi * np.sin(np.radians(t / 2)) / L1


def fp_screen(db_entries, obs, tol=0.12, shifts=np.arange(-0.18, 0.1801, 0.06), epss=np.arange(-0.012, 0.01201, 0.003)):
    """Vectorised peak-list FOM for every entry over (2θ shift × lattice scale); returns best score/shift/eps per entry."""
    t_obs, I_obs = obs["t"], obs["I"]; lo, hi = obs["range"]
    E = len(db_entries)
    idx, dd, II, Imax = [], [], [], np.zeros(E)
    for k, e in enumerate(db_entries):
        m = np.isfinite(e["t"]) & (e["t"] >= lo - 2) & (e["t"] <= hi + 2) & (e["I"] >= 2)
        if m.sum():
            idx.append(np.full(m.sum(), k)); dd.append(e["d"][m]); II.append(e["I"][m])
    idx, dd, II = np.concatenate(idx), np.concatenate(dd), np.concatenate(II)
    p_rand = min(0.9, len(t_obs) * 2 * tol / max(1.0, hi - lo))
    best = np.zeros(E); bs = np.zeros(E); be = np.zeros(E); bcov = np.zeros(E); bn = np.zeros(E)
    for eps in epss:
        sn = 1.540598 / (2 * dd * (1 + eps))
        tt = np.where(sn < 1, 2 * np.degrees(np.arcsin(np.clip(sn, 0, 1))), np.inf)
        inr = (tt >= lo + 0.15) & (tt <= hi - 0.15)
        Imx = np.zeros(E); np.maximum.at(Imx, idx[inr], II[inr])
        Irel = np.where(inr, II / np.maximum(Imx[idx], 1e-9), 0)
        Iw = np.sqrt(Irel)                           # flatter weights: less sensitive to preferred orientation
        tot = np.bincount(idx, Iw, E)
        strong = inr & (Irel >= 0.35)
        n_strong = np.bincount(idx, strong, E)
        for s in shifts:
            pos = tt + s
            j = np.clip(np.searchsorted(t_obs, pos), 1, len(t_obs) - 1)
            d = np.minimum(np.abs(t_obs[j - 1] - pos), np.abs(t_obs[j] - pos))
            ok = inr & (d <= tol)
            cov = np.bincount(idx, Iw * ok, E) / np.maximum(tot, 1e-9)
            cs = np.bincount(idx, strong & ok, E) / np.maximum(n_strong, 1)
            n = np.bincount(idx, ok & (Irel >= 0.05), E)
            err = np.bincount(idx, np.where(ok, d, 0), E) / np.maximum(np.bincount(idx, ok, E), 1) / tol
            sc = cov * cs * (1 - 0.3 * err) * (1 - p_rand ** n) * (1 - 0.15 * abs(eps) / 0.012)
            sc[(Imx < 15) | (n < 2)] = 0
            upd = sc > best
            best[upd] = sc[upd]; bs[upd] = s; be[upd] = eps; bcov[upd] = cov[upd]; bn[upd] = n[upd]
    return best, bs, be, bcov, bn


def fp_pv_kernel(n, dq, fwhm, eta=0.5):
    x = (np.arange(n) - n // 2) * dq
    g = np.exp(-4 * math.log(2) * x * x / fwhm ** 2)
    l = 1 / (1 + 4 * x * x / fwhm ** 2)
    k = eta * l + (1 - eta) * g
    return np.fft.ifftshift(k / k.sum())


def fp_estimate_width(q, yb, obs_t):
    """median FWHM (in q) of the strongest, reasonably isolated peaks"""
    ws = []
    order = np.argsort(-yb[np.searchsorted(q, fp_to_q(obs_t)).clip(0, len(q) - 1)])
    for i in order[:12]:
        c = int(np.searchsorted(q, fp_to_q(obs_t[i])))
        c = max(1, min(len(q) - 2, c))
        h = yb[c]
        if h <= 0: continue
        a = c
        while a > 0 and yb[a] > h / 2: a -= 1
        b = c
        while b < len(q) - 1 and yb[b] > h / 2: b += 1
        if 2 < b - a < 200: ws.append(q[b] - q[a])
    return float(np.median(ws)) if ws else 0.01


class FPFitter:
    def __init__(self, x, y, kalpha2, bg_deg=5):
        o = np.argsort(x); x, y = x[o], y[o]
        self.x, self.y = x, y
        q = fp_to_q(x)
        n = len(x)
        self.q = np.linspace(q.min(), q.max(), n)
        self.dq = self.q[1] - self.q[0]
        yq = np.interp(self.q, q, y)
        step = float(np.median(np.diff(x)))
        self.bg = snip_background(yq, max(2, int(round(3.0 / step / 2))))
        self.yb = yq - self.bg
        self.yq = yq
        self.set_weight_power(0.5)
        u = 2 * (self.q - self.q[0]) / (self.q[-1] - self.q[0]) - 1
        self.B = np.stack([np.polynomial.legendre.legval(u, np.eye(bg_deg + 1)[k]) for k in range(bg_deg + 1)], 1)
        self.kalpha2 = kalpha2
        self.M = 1 << int(math.ceil(math.log2(n * 2)))
        self.n = n
        self._prep()

    def set_weight_power(self, p):
        """weights 1/y^p: p = 0.5 counting statistics (search), smaller p for the final intensity shares"""
        floor = max(1e-9, 0.01 * float(np.median(np.abs(self.yq))))
        self.w = np.maximum(self.yq, floor) ** (-p)
        self.w /= self.w.mean()
        if hasattr(self, "B"):
            self._prep()

    def _prep(self):
        self.Bw = self.B * self.w[:, None]
        self.bw = self.yb * self.w
        self.bb = float(self.bw @ self.bw)
        self.GB = self.Bw.T @ self.Bw
        self.rB = self.Bw.T @ self.bw

    def set_widths(self, w0, mults=(0.8, 1.0, 1.4, 2.0, 3.0, 5.0)):
        self.mults = mults
        self.K = [np.fft.rfft(fp_pv_kernel(self.M, self.dq, w0 * m)) for m in mults]

    def profiles(self, e, shift, eps=0.0, sel=None):
        """profiles (len(mults) x n) of entry e with 2θ shift and lattice scale (1+eps); sel = mask of lines to include"""
        m = np.isfinite(e["t"]) & (e["I"] >= 1)
        if sel is not None:
            m &= sel
        d = e["d"][m] * (1 + eps); I = e["I"][m] / 100.0
        sn = L1 / (2 * d); ok = sn < 1
        t = 2 * np.degrees(np.arcsin(sn[ok])) + shift; I = I[ok]
        qs = fp_to_q(t)
        pos = [(qs, I)]
        if self.kalpha2:
            pos.append((qs * L1 / L2, 0.5 * I))
        S = np.zeros(self.M)
        for qq, ii in pos:
            f = (qq - self.q[0]) / self.dq
            ok = (f >= 0) & (f < self.n - 1)
            i0 = np.floor(f[ok]).astype(int); fr = f[ok] - i0
            np.add.at(S, i0, ii[ok] * (1 - fr)); np.add.at(S, i0 + 1, ii[ok] * fr)
        F = np.fft.rfft(S)
        return np.stack([np.fft.irfft(F * K, self.M)[: self.n] for K in self.K])

    def solve(self, P):
        """weighted LSQ: yb ≈ B c + P^T a, a >= 0 (normal equations; active set by dropping negatives)"""
        nb = self.B.shape[1]
        Pw = P * self.w[None, :] if len(P) else np.zeros((0, self.n))
        G_pp = Pw @ Pw.T; G_bp = self.Bw.T @ Pw.T; r_p = Pw @ self.bw
        cols = list(range(P.shape[0]))
        while True:
            k = len(cols)
            G = np.empty((nb + k, nb + k)); G[:nb, :nb] = self.GB
            if k:
                G[:nb, nb:] = G_bp[:, cols]; G[nb:, :nb] = G_bp[:, cols].T; G[nb:, nb:] = G_pp[np.ix_(cols, cols)]
            rhs = np.concatenate([self.rB, r_p[cols]])
            G[np.diag_indices_from(G)] += 1e-12 * (np.trace(G) / len(G) + 1e-30)
            try:
                coef = np.linalg.solve(G, rhs)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(G, rhs, rcond=None)[0]
            a = coef[nb:]
            if (a >= 0).all() or not cols:
                full = np.zeros(P.shape[0]); full[cols] = a
                ssr = max(0.0, self.bb - 2 * coef @ rhs + coef @ G @ coef)
                return full, float(ssr), coef[:nb]
            cols = [c for c, v in zip(cols, a) if v > 0]


def fp_parse_hkl(s):
    s = str(s).strip()
    v = s.split() if " " in s else re.findall(r"-?\d", s)
    try:
        v = [int(a) for a in v]
    except ValueError:
        return None
    return v if len(v) == 3 else None


def fp_po_family(e, lo, hi):
    """axial reflection family (h00 / 0k0 / 00l) that may be enhanced by preferred orientation of plates;
    returns a boolean mask over e's lines, or None"""
    key = (round(lo, 1), round(hi, 1))
    cache = e.setdefault("_po", {})
    if key in cache:
        return cache[key]
    hk = [fp_parse_hkl(h) for h in e["hkl"]]
    inr = np.isfinite(e["t"]) & (e["t"] >= lo) & (e["t"] <= hi)
    top5 = set(np.argsort(-np.where(inr, e["I"], -1))[:5])
    best, bestI = None, 0
    for ax in range(3):
        mask = np.array([h is not None and all(h[i] == 0 for i in range(3) if i != ax) and h[ax] != 0 for h in hk])
        if (mask & inr & (e["I"] >= 3)).sum() >= 2 and any(mask[i] for i in top5):
            tot = e["I"][mask & inr].sum()
            if tot > bestI: best, bestI = mask, tot
    cache[key] = best
    return best


def fp_line_set(e, lo, hi, shift=0.0, imin=10):
    m = np.isfinite(e["t"]) & (e["t"] >= lo) & (e["t"] <= hi) & (e["I"] >= imin)
    return e["t"][m] + shift, e["I"][m]


def fp_similarity(a, b, lo, hi):
    """peak-pattern fp_similarity of two entries (0–1): share of strong-line intensity that coincides, both ways"""
    ta, Ia = fp_line_set(a, lo, hi); tb, Ib = fp_line_set(b, lo, hi)
    if len(ta) == 0 or len(tb) == 0: return 0.0
    best = 0.0
    for s in np.arange(-0.3, 0.301, 0.03):
        da = np.min(np.abs(ta[:, None] - (tb[None, :] + s)), 1) <= 0.12
        db = np.min(np.abs(tb[:, None] + s - ta[None, :]), 1) <= 0.12
        best = max(best, min((Ia * da).sum() / Ia.sum(), (Ib * db).sum() / Ib.sum()))
    return best


NON_MINERAL_ELEMENTS = set("""hydrogen helium lithium beryllium boron nitrogen oxygen fluorine neon sodium magnesium
aluminum aluminium phosphorus chlorine argon potassium calcium scandium titanium vanadium manganese gallium germanium
bromine krypton rubidium strontium yttrium zirconium niobium molybdenum technetium cesium caesium barium lanthanum
cerium praseodymium neodymium promethium samarium europium gadolinium terbium dysprosium holmium erbium thulium
ytterbium lutetium hafnium rhenium thallium polonium astatine radon francium radium actinium thorium protactinium
uranium neptunium plutonium americium curium berkelium californium einsteinium xenon iodine carbon""".split())


def is_mineral_entry(e):
    b = base_name(e["name"]).split("-")[0]
    high_p = bool(re.search(r"\s[IVX]+$", e["name"].strip()))      # e.g. Ice III, Ice VII (high-pressure phases)
    return not (e["src"] == "A" and (looks_synthetic(e["name"]) or b in NON_MINERAL_ELEMENTS or high_p))


COMMON_MINERALS = set("""quartz cristobalite tridymite opal calcite aragonite vaterite dolomite ankerite siderite magnesite rhodochrosite
smithsonite cerussite strontianite witherite albite oligoclase andesine labradorite bytownite anorthite microcline orthoclase
sanidine anorthoclase nepheline leucite analcime muscovite phengite paragonite illite biotite phlogopite annite lepidolite
glauconite celadonite chlorite clinochlore chamosite kaolinite halloysite dickite nacrite montmorillonite nontronite beidellite
saponite vermiculite palygorskite sepiolite talc pyrophyllite antigorite lizardite chrysotile tremolite actinolite hornblende
pargasite glaucophane riebeckite augite diopside hedenbergite aegirine jadeite omphacite pigeonite enstatite ferrosilite
wollastonite forsterite fayalite almandine pyrope grossular andradite spessartine epidote clinozoisite zoisite allanite titanite
zircon rutile anatase brookite ilmenite magnetite maghemite hematite goethite lepidocrocite ferrihydrite akaganeite gibbsite
boehmite diaspore corundum spinel chromite pyrite marcasite pyrrhotite chalcopyrite galena sphalerite bornite chalcocite
covellite arsenopyrite gypsum bassanite anhydrite baryte celestine jarosite alunite natroalunite halite sylvite thenardite
mirabilite epsomite hexahydrite fluorite fluorapatite hydroxylapatite chlorapatite apatite monazite xenotime kyanite
sillimanite andalusite staurolite cordierite schorl dravite elbaite tourmaline beryl topaz mullite brucite periclase
portlandite ettringite hydrotalcite clinoptilolite heulandite chabazite mordenite stilbite phillipsite natrolite laumontite
prehnite pumpellyite vesuvianite cassiterite scheelite wolframite malachite azurite cuprite sulfur gold copper silver""".split())


MINERAL_GROUPS = {
    "斜長石 plagioclase": "albite oligoclase andesine labradorite bytownite anorthite",
    "鹼性長石 alkali feldspar": "microcline orthoclase sanidine anorthoclase adularia",
    "石榴子石 garnet": "almandine pyrope spessartine grossular andradite uvarovite hibschite katoite majorite",
    "單斜輝石 clinopyroxene": "diopside hedenbergite augite aegirine jadeite omphacite pigeonite petedunnite johannsenite esseneite",
    "斜方輝石 orthopyroxene": "enstatite ferrosilite clinoenstatite protoenstatite",
    "橄欖石 olivine": "forsterite fayalite tephroite monticellite",
    "磷灰石 apatite": "fluorapatite hydroxylapatite chlorapatite apatite carbonate-fluorapatite carbonate-hydroxylapatite",
    "高嶺石族 kaolin": "kaolinite dickite nacrite halloysite",
    "膨潤石 smectite": "montmorillonite nontronite beidellite saponite hectorite",
    "綠泥石 chlorite": "clinochlore chamosite chlorite pennantite nimite",
    "二八面體雲母 dioctahedral mica": "muscovite phengite illite paragonite celadonite glauconite margarite",
    "三八面體雲母 trioctahedral mica": "biotite phlogopite annite siderophyllite lepidolite polylithionite eastonite fluorophlogopite",
    "角閃石 amphibole": "tremolite actinolite ferro-actinolite hornblende magnesio-hornblende ferrohornblende pargasite edenite",
    "方解石族 calcite group": "calcite magnesite siderite rhodochrosite smithsonite otavite",
    "白雲石族 dolomite group": "dolomite ankerite kutnohorite minrecordite",
    "霰石族 aragonite group": "aragonite strontianite witherite cerussite",
    "綠簾石族 epidote group": "epidote clinozoisite zoisite piemontite allanite",
    "蛇紋石 serpentine": "antigorite lizardite chrysotile",
    "鋁矽酸鹽 Al2SiO5": "kyanite sillimanite andalusite mullite",
}
MINERAL_GROUP_OF = {m: g for g, ms in MINERAL_GROUPS.items() for m in ms.split()}


def fp_group_of(e):
    b = base_name(e["name"]).split()[0]
    return MINERAL_GROUP_OF.get(b) or MINERAL_GROUP_OF.get(b.split("-")[0])


def fp_is_common(e):
    b = base_name(e["name"]).split()[0]
    return b in COMMON_MINERALS or b.split("-")[-1] in COMMON_MINERALS or b.split("-")[0] in COMMON_MINERALS


def fp_prior(e):
    """common rock / soil / ore minerals 1.0; others 0.75–0.9 by number of RRUFF (natural) samples"""
    return 1.0 if fp_is_common(e) else 0.75 + 0.05 * min(e.get("n_r", 0), 3)


def fp_is_rare(e):
    return not fp_is_common(e) and e.get("n_r", 0) <= 1


def full_identify(x, y, db_entries, kalpha2, allowed=None, required=None, pool_size=400, max_phases=12,
             t_possible=0.003, t_certain=0.015, minerals_only=True, verbose=False):
    obs = detect_peaks(x, y)
    obs_lo = detect_peaks_sensitive(x, y)
    lo, hi = obs_lo["range"]
    ents = [e for e in db_entries
            if (not (allowed or required) or element_ok(e["el"], allowed or set(), required or set()))
            and (not minerals_only or is_mineral_entry(e))]
    if not ents:
        return dict(phases=[], rwp=1.0, pool=0, fit_t=np.array([]), fit_obs=np.array([]), fit_calc=np.array([]),
                    fit_bg=np.array([]))
    sc, sh, ep, cov, nmatch = fp_screen(ents, obs)
    order = np.argsort(-sc)
    pool, seen = [], {}
    for k in order:
        if sc[k] <= 0.05: break
        nm = base_name(ents[k]["name"])
        if seen.get(nm, 0) >= 2: continue
        seen[nm] = seen.get(nm, 0) + 1
        pool.append(int(k))
        if len(pool) >= pool_size: break
    F = FPFitter(x, y, kalpha2)
    F.set_widths(fp_estimate_width(F.q, F.yb, obs["t"]))
    par = {k: [sh[k], ep[k]] for k in pool}                       # 2θ shift, lattice scale
    po = {k: fp_po_family(ents[k], lo, hi) for k in pool}           # oriented reflection family (or None)
    prof, profb = {}, {}

    def make(k, shift, eps):
        P = F.profiles(ents[k], shift, eps)
        Pb = F.profiles(ents[k], shift, eps, po[k]) if po[k] is not None else None
        return P, Pb

    for k in pool:
        prof[k], profb[k] = make(k, *par[k])
    t_lo = obs_lo["t"]
    ssr0 = float((F.yb * F.w) @ (F.yb * F.w))
    sel, sel_w, tiers, evid = [], [], {}, {}

    PO_MAX = 4.0   # oriented reflections may be enhanced or weakened at most 4× (≈ March–Dollase r 0.6–1.6)

    def build(ks, ws, over=None, fixed=None):
        rows, owner, kind = [], [], []
        for i, (k, w) in enumerate(zip(ks, ws)):
            if over is not None and k in over:
                pf, pb = over[k]
            else:
                pf = prof[k][w]; pb = profb[k][w] if profb[k] is not None else None
            if pb is None:
                rows.append(pf); owner.append(i); kind.append(0)
            elif fixed and i in fixed:
                rows.append(pf - pb + fixed[i] * pb); owner.append(i); kind.append(0)
            else:
                rows.append(pf - pb); owner.append(i); kind.append(1); rows.append(pb); owner.append(i); kind.append(2)
        return (np.stack(rows) if rows else np.zeros((0, F.n))), np.array(owner, int), np.array(kind, int)

    def fit(ks, ws, over=None):
        P, own, kind = build(ks, ws, over)
        a, ss, bgc = F.solve(P)
        fixed = {}
        for i in range(len(ks)):
            r_i = np.where((own == i) & (kind == 1))[0]; p_i = np.where((own == i) & (kind == 2))[0]
            if len(r_i) and len(p_i):
                ar, ap = a[r_i[0]], a[p_i[0]]
                if ap > PO_MAX * ar: fixed[i] = PO_MAX
                elif ap < ar / PO_MAX: fixed[i] = 1 / PO_MAX
        if fixed:
            P, own, kind = build(ks, ws, over, fixed)
            a, ss, bgc = F.solve(P)
        comps = [a[own == i] @ P[own == i] for i in range(len(ks))]
        return ss, comps, bgc

    def lines_of(k, imin=5):
        e = ents[k]; d = e["d"] * (1 + par[k][1]); sn = L1 / (2 * d)
        t = np.where(sn < 1, 2 * np.degrees(np.arcsin(np.clip(sn, 0, 1))), np.nan) + par[k][0]
        m = np.isfinite(t) & (t >= lo + 0.15) & (t <= hi - 0.15) & (e["I"] >= imin)
        return t[m], e["I"][m]

    q_lo = fp_to_q(t_lo) if len(t_lo) else np.array([])
    i_lo = np.clip(np.round((q_lo - F.q[0]) / F.dq).astype(int), 0, F.n - 1) if len(t_lo) else np.array([], int)

    def evidence(k, comp_k, comp_others):
        """top-line presence; observed peaks this phase 'owns' in the fit (≥ 50 % of the model there);
        owned share of the phase's own fitted intensity"""
        t, I = lines_of(k, 5)
        if len(t) == 0 or not len(t_lo): return (0, 0), 0, 0.0
        near = np.min(np.abs(t[:, None] - t_lo[None, :]), 1) <= 0.12
        top = np.argsort(-I)[:3]
        top_ok = (int(near[top].sum()), len(top))
        ck, co = comp_k[i_lo], comp_others[i_lo]
        on_line = np.min(np.abs(t_lo[:, None] - t[None, :]), 1) <= 0.12
        owned = (ck > 0.5 * (ck + co) + 1e-30) & (F.yb[i_lo] > 0) & on_line & (ck > 0.25 * np.maximum(F.yb[i_lo], 1e-30))
        uI = float(ck[owned].sum() / max(ck[on_line].sum(), 1e-30))
        return top_ok, int(owned.sum()), uI

    def refine(i):
        """per-phase 2θ shift / lattice scale / width refinement against data with the other phases"""
        k = sel[i]; best = None
        for eps in par[k][1] + np.arange(-0.003, 0.00301, 0.0015):
            for ds in (-0.04, 0.0, 0.04):
                P, Pb = make(k, par[k][0] + ds, eps)
                for w in range(len(F.mults)):
                    ws = list(sel_w); ws[i] = w
                    ss, _, _ = fit(sel, ws, {k: (P[w], Pb[w] if Pb is not None else None)})
                    if best is None or ss < best[0]: best = (ss, eps, ds, w, P, Pb)
        ss, eps, ds, w, P, Pb = best
        par[k] = [par[k][0] + ds, eps]; prof[k] = P; profb[k] = Pb; sel_w[i] = w
        return ss

    ssr, _, _ = fit(sel, sel_w)
    for step in range(max_phases):
        _, comps, bgc = fit(sel, sel_w)
        model = F.B @ bgc + (np.sum(comps, 0) if comps else 0)
        r = (F.yb - model) * F.w
        gains = []
        for k in pool:
            if k in sel: continue
            Pw = prof[k] * F.w
            num = Pw @ r; den = (Pw * Pw).sum(1) + 1e-30
            g = np.where(num > 0, num * num / den, 0)
            j = int(np.argmax(g)); gains.append((g[j], k, j))
        gains.sort(reverse=True)
        best = None
        for g, k, j in gains[:20]:
            gk = fp_group_of(ents[k])
            if any(fp_similarity(ents[k], ents[kk], lo, hi) >= (0.5 if gk and gk == fp_group_of(ents[kk]) else 0.75)
                   for kk in sel): continue   # same structure family / same solid-solution series
            ss, comps, _ = fit(sel + [k], sel_w + [j])
            comp_k = comps[-1]
            if comp_k.sum() <= 0: continue
            dssr = (ssr - ss) / ssr0
            comp_o = np.sum(comps[:-1], 0) if sel else np.zeros(F.n)
            (tk, tn), uniq, uI = evidence(k, comp_k, comp_o)
            rare = fp_is_rare(ents[k])
            ok_possible = (dssr >= t_possible * (2 if rare else 1) and tk >= min(2, tn) and uI >= 0.10
                           and uniq >= (3 if rare else 2))
            if not ok_possible: continue
            rank = dssr * fp_prior(ents[k]) * (1 - 0.3 * abs(par[k][1]) / 0.012) * (0.4 + 0.6 * sc[k])
            if best is None or rank > best[0]: best = (rank, k, j, ss, uniq, uI, tk, tn, dssr)
        if best is None: break
        _, k, j, ss, uniq, uI, tk, tn, dssr = best
        sel.append(k); sel_w.append(j)
        ssr = refine(len(sel) - 1)
        certain = (dssr >= t_certain and uniq >= 3 and uI >= 0.25 and tk == tn and sc[k] >= 0.45
                   and (not fp_is_rare(ents[k]) or dssr >= 3 * t_certain))
        tiers[k] = "certain" if certain else "possible"
        evid[k] = dict(dssr=dssr, uniq=uniq, uI=uI, top=f"{tk}/{tn}")
        if verbose: print(f"  + {ents[k]['name']:<18} dSSR {dssr*100:5.2f}%  uniq {uniq:2d}  uniqI {uI:.2f}  top {tk}/{tn}  {tiers[k]}{'  PO' if po[k] is not None else ''}")
    for i in range(len(sel)):
        ssr = refine(i)
    # backward: drop phases whose removal barely matters
    changed = True
    while changed and len(sel) > 1:
        changed = False
        full, _, _ = fit(sel, sel_w)
        for i in range(len(sel)):
            s2, _, _ = fit(sel[:i] + sel[i + 1:], sel_w[:i] + sel_w[i + 1:])
            if (s2 - full) / ssr0 < t_possible * 0.5:
                if verbose: print("  - drop", ents[sel[i]]["name"])
                del sel[i]; del sel_w[i]; changed = True; break
    ssr, comps, bgc = fit(sel, sel_w)
    rwp = math.sqrt(ssr / ssr0)
    F.set_weight_power(0.15)          # final scale factors: closer to an unweighted fit, so strong peaks count fully
    _, comps, bgc = fit(sel, sel_w)
    calc = F.bg + F.B @ bgc + (np.sum(comps, 0) if comps else 0)
    areas = np.array([max(0.0, c.sum()) for c in comps])
    tot = areas.sum() if len(areas) and areas.sum() > 0 else 1.0
    out = []
    for i, (k, w) in enumerate(zip(sel, sel_w)):
        cands = []
        for kk in pool:
            if kk in sel or base_name(ents[kk]["name"]) == base_name(ents[k]["name"]): continue
            cands.append((fp_similarity(ents[k], ents[kk], lo, hi), kk))
        cands.sort(reverse=True)
        alts = []
        for sim, kk in cands[:8]:
            if sim < 0.6: break
            bestss = None
            for eps in par[kk][1] + np.arange(-0.006, 0.00601, 0.003):
                P, Pb = make(kk, par[kk][0], eps)
                for jj in sorted({max(0, w - 1), w, min(len(F.mults) - 1, w + 1)}):
                    ks = list(sel); ks[i] = kk; ws = list(sel_w); ws[i] = jj
                    s2, _, _ = fit(ks, ws, {kk: (P[jj], Pb[jj] if Pb is not None else None)})
                    bestss = s2 if bestss is None else min(bestss, s2)
            # same structure family is always listed; others only if they fit almost as well
            if sim >= 0.75 or (bestss - ssr) / ssr0 < 0.015:
                alts.append(((bestss - ssr) / ssr0, ents[kk]["name"]))
        g = fp_group_of(ents[k])
        grp = []
        if g:   # other members of the same solid-solution series that are chemically allowed and in the candidate pool
            grp = sorted({ents[kk]["name"] for kk in pool if kk not in sel and fp_group_of(ents[kk]) == g
                          and base_name(ents[kk]["name"]) != base_name(ents[k]["name"])})
        alts = list(dict.fromkeys([n for _, n in sorted(alts)] + grp))[:6]
        out.append(dict(name=ents[k]["name"], key=ents[k]["key"], tier=tiers.get(k, "possible"),
                        share=float(areas[i] / tot), shift=float(par[k][0]), eps=float(par[k][1]),
                        width=F.mults[w], alts=alts, lines=lines_of(k, 5)[0], po=po[k] is not None, group=g,
                        **evid.get(k, {})))
    out.sort(key=lambda d: -d["share"])
    tq = 2 * np.degrees(np.arcsin(np.clip(F.q * L1 / (4 * np.pi), 0, 1)))
    step = max(1, F.n // 1500)
    return dict(phases=out, rwp=rwp, pool=len(pool), fit_t=tq[::step],
                fit_obs=(F.yb + F.bg)[::step], fit_calc=calc[::step], fit_bg=F.bg[::step])


def detect_peaks_sensitive(x, y):
    """more sensitive peak list (4σ, 0.15 % of max) for screening and counting unique peaks"""
    step = float(np.median(np.diff(x))) or 0.02
    yn = y - snip_background(y, max(1, min(int(round(1.5 / step / 2)), len(y) // 2 - 1)))
    noise = 1.4826 * float(np.median(np.abs(np.diff(yn)))) / math.sqrt(2) or 1e-12
    w = max(2, int(round(0.06 / step)))
    win = _sliding_max(yn, w)
    cand = np.where((yn >= win) & (yn > 4 * noise) & (yn > 0.0015 * yn.max()))[0]
    keep, last = [], -10 ** 9
    for i in cand:
        if i - last > w: keep.append(i); last = i
    t = x[np.array(keep, int)] if keep else np.array([])
    I = 100 * yn[keep] / yn[keep].max() if keep else np.array([])
    return dict(t=t, I=I, range=(float(x.min()), float(x.max())))


TICK_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#8C564B",
               "#7F7F7F", "#BCBD22", "#17BECF", "#E377C2"]


def fit_png(t, obs, calc, bg, phases, width_px=900, dpi=150):
    """t: 2θ grid; obs/calc/bg: arrays; phases: list of (name, tier, line_positions_2θ)"""
    n = len(phases)
    W = width_px / dpi
    H = W * (0.52 + 0.035 * n)
    fig = Figure(figsize=(W, H), dpi=dpi)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.25 + 0.16 * n, 0.9], hspace=0.0,
                          left=0.03, right=0.985, top=0.985, bottom=0.11)
    ax = fig.add_subplot(gs[0]); at = fig.add_subplot(gs[1], sharex=ax); ar = fig.add_subplot(gs[2], sharex=ax)
    with matplotlib.rc_context({"font.size": 7}):
        ax.plot(t, obs, color="#8e8e93", lw=0.9, label="obs")
        ax.plot(t, calc, color="#E64B35", lw=0.7, label="fit")
        ax.plot(t, bg, color="#c7c7cc", lw=0.6, ls="--")
        for i, (name, tier, pos) in enumerate(phases):
            c = TICK_COLORS[i % len(TICK_COLORS)]
            y = n - i
            at.vlines(pos, y - 0.35, y + 0.35, color=c, lw=0.7)
            at.text(1.0, y, " " + name, transform=at.get_yaxis_transform(), ha="right", va="center", fontsize=6.5,
                    color=c, fontweight="bold" if tier == "certain" else "normal",
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
        at.set_ylim(0.3, n + 0.7)
        res = obs - calc
        ar.plot(t, res, color="#3C5488", lw=0.6)
        ar.axhline(0, color="#c7c7cc", lw=0.5)
        lim = max(np.percentile(np.abs(res), 99.5), 1e-9) * 1.2
        ar.set_ylim(-lim, lim)
        for a in (ax, at, ar):
            a.set_yticks([])
            a.tick_params(axis="x", direction="in", labelsize=7, length=2.5, width=0.5)
            for sp in a.spines.values():
                sp.set_linewidth(0.5)
        ax.tick_params(labelbottom=False); at.tick_params(labelbottom=False)
        ar.set_xlabel("2θ (°)", fontsize=7, labelpad=1)
        ax.set_xlim(t.min(), t.max())
        ax.text(0.005, 0.95, "grey: observed   red: fit", transform=ax.transAxes, fontsize=6.5, va="top", color="#6e6e73")
        ar.text(0.005, 0.9, "residual", transform=ar.transAxes, fontsize=6.5, va="top", color="#6e6e73")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    return buf.getvalue()


SENSITIVITY = {"high": dict(t_possible=0.003, t_certain=0.015), "std": dict(t_possible=0.006, t_certain=0.02)}


def id_auto(S, items, a):
    import time as _time
    t0 = _time.time()
    iid = a.get("id")
    if iid not in DATA:
        raise UserError("請先選擇要鑑定的曲線。")
    it = next((i for i in items if i.get("id") == iid), {})
    d0 = DATA[iid]
    eff = effective(d0, S)
    x, y, _ = converted_xy(d0, eff)
    x = x + (fnum(it.get("shift"), 0.0) or 0.0)
    o = np.argsort(x)
    x, y = x[o], y[o]
    allowed = {s for s in re.split(r"[\s,;]+", a.get("allowed") or "") if s in ELEMENTS} or None
    required = {s for s in re.split(r"[\s,;]+", a.get("required") or "") if s in ELEMENTS} or None
    kalpha2 = eff["type"] == "inhouse"   # 實驗室 Cu 光源有 Kα2；同步輻射為單一波長
    par = SENSITIVITY.get(a.get("sensitivity") or "high", SENSITIVITY["high"])
    res = full_identify(x, y, mindb()["e"], kalpha2, allowed, required, minerals_only=a.get("minerals_only", True),
                        **par)
    by_key = mindb()["by_key"]
    phases = []
    for p in res["phases"]:
        e = by_key[p["key"]]
        phases.append(dict(entry_info(e), tier=p["tier"], share=round(p["share"], 4), shift=round(p["shift"], 3),
                           eps=round(p["eps"], 4), alts=p["alts"], dssr=round(p.get("dssr", 0), 4),
                           uniq=p.get("uniq", 0), group=p.get("group"), po=bool(p.get("po"))))
    png = fit_png(res["fit_t"], res["fit_obs"], res["fit_calc"], res["fit_bg"],
                  [(p["name"], p["tier"], p["lines"]) for p in res["phases"]])
    return dict(phases=phases, rwp=round(res["rwp"], 4), pool=res["pool"], kalpha2=kalpha2,
                secs=round(_time.time() - t0, 1)), png


# ═════════════════════════════════════════════════════════════════════════════
# 11. 給網頁呼叫的入口：api(指令, JSON 參數) → (JSON 結果, 檔案位元組或 None)
# ═════════════════════════════════════════════════════════════════════════════
def _api(cmd, a):
    S_in, items = a.get("S") or {}, a.get("items") or []
    if cmd == "upload":
        metas, notes, new_ids = [], [], set()
        trim = fnum(a.get("trim"), DEFAULTS["csv_trim"])
        for f in a.get("files") or []:
            try:
                d = load_dataset(str(f.get("name") or "data.txt"), str(f.get("text") or ""), trim)
            except UserError as e:
                notes.append(dict(level="warn", ids=[], text=str(e)))
                continue
            DATA[d["id"]] = d
            new_ids.add(d["id"])
            metas.append(dataset_meta(d))
        notes += verify_pairs(new_ids, fnum(a.get("src_wl"), ORI_WL))
        return {"datasets": metas, "notes": notes}, None
    if cmd == "remove":
        for i in a.get("ids") or []:
            DATA.pop(i, None)
        return {"ok": True}, None
    if cmd == "render":
        S = merge_settings(S_in)
        dpi = max(40.0, min(400.0, fnum(a.get("px"), 900) / (max(20.0, S["width_mm"]) / 25.4)))
        data, info, warnings, size, _ = render_bytes(S_in, items, dpi, "png", False)
        return dict(warnings=warnings, size=size, **info), data
    if cmd == "peaks":
        S = merge_settings(S_in)
        P = prepare(S, items, [])
        c = next((c for c in P["pre"] if c["id"] == a.get("id")), None)
        if c is None:
            raise UserError("找不到要偵測的曲線（請確認它是顯示狀態）。")
        m = (c["xv"] >= P["view"][0]) & (c["xv"] <= P["view"][1])
        return {"peaks": find_peaks_simple(c["xv"][m], c["yv"][m], int(fnum(a.get("n"), 10))), "idx": c["idx"]}, None
    if cmd == "export_figure":
        name, data, mime = export_figure(S_in, items)
        return {"name": name, "mime": mime}, data
    if cmd == "export_csv":
        S = merge_settings(S_in)
        only = a.get("only")
        files = all_converted_csv(S, [it for it in items if not only or it.get("id") == only])
        if not files:
            raise UserError("沒有可匯出的檔案。")
        if len(files) == 1:
            return {"name": files[0][0], "mime": "text/csv"}, files[0][1].encode("utf-8")
        return {"name": "XRD_converted_csv.zip", "mime": "application/zip"}, make_zip(files)
    if cmd == "export_all":
        S = merge_settings(S_in)
        name, data, _ = export_figure(S_in, items)
        files = [(name, data)] + [("csv/" + n, t) for n, t in all_converted_csv(S, items)]
        files.append(("settings.json", json.dumps(dict(settings=S_in, items=items, version=APP_VERSION,
                                                       exported=datetime.now().isoformat(timespec="seconds")),
                                                  ensure_ascii=False, indent=2)))
        return {"name": safe_name(S["fig_name"]) + "_export.zip", "mime": "application/zip"}, make_zip(files)
    if cmd == "id_peaks":
        return id_peaks(merge_settings(S_in), items, a), None
    if cmd == "id_search":
        return id_search(merge_settings(S_in), items, a), None
    if cmd == "id_verify":
        return id_verify(merge_settings(S_in), items, a), None
    if cmd == "id_auto":
        return id_auto(merge_settings(S_in), items, a)
    if cmd == "id_ref":
        return id_ref(a), None
    if cmd == "id_names":
        db = mindb()
        return dict(names=db["names"], version=db["version"], cite=db["cite"], n=len(db["e"])), None
    if cmd == "info":
        return {"version": APP_VERSION, "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                "python": sys.version.split()[0], "fonts": sorted(available_fonts() & {
                    f for s in FONT_STACKS.values() for f in s})}, None
    raise UserError("未知的指令：%s" % cmd)


def api(cmd, args_json):
    try:
        meta, data = _api(cmd, json.loads(args_json or "{}"))
    except UserError as e:
        meta, data = {"error": str(e)}, None
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        meta, data = {"error": "%s: %s" % (type(e).__name__, e)}, None
    if data is not None:
        try:  # Pyodide：直接轉成 JavaScript 的 Uint8Array
            from pyodide.ffi import to_js
            data = to_js(memoryview(data))
        except ImportError:
            pass
    return json.dumps(meta, ensure_ascii=False), data


# ═════════════════════════════════════════════════════════════════════════════
# 12. 本機批次轉換（與原 Synchrotron_Convert.py 相同的輸出，不需開介面）
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python xrd_core.py 檔案1.xye [檔案2 ...] → 在原檔旁輸出換算後的 CSV")
        sys.exit(0)
    S = merge_settings({})
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8", errors="replace") as f:
            d = load_dataset(os.path.basename(path), f.read(), S["csv_trim"])
        DATA[d["id"]] = d
        name, text = all_converted_csv(S, [{"id": d["id"]}])[-1]
        out = os.path.join(os.path.dirname(os.path.abspath(path)), name)
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print("Processed and saved file:", out)
