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

APP_VERSION = "2.8"

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
    "first_on_top": True, "y_axes": "shared",
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
        if S["y_axes"] == "each":
            # 各自的 Y 軸：每條曲線在顯示範圍內的最低點對齊底部、最高點對齊頂部
            # （等同每條曲線各有一個自動縮放、不顯示的 Y 軸；此時「正規化」不再有作用）
            mv = (x >= x0) & (x <= x1)
            ys = y[mv] if mv.sum() >= 2 else y
            lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys))
            y = (y - lo) / (hi - lo) if hi > lo else y - lo
        else:
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
        text = (parts[1] if len(parts) > 1 else "").replace("\\n", "\n")   # 「\n」= 換行
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
        rir = row[8] if len(row) > 8 else None
        rows = [tok.split(":", 2) for tok in pk.split(";") if tok]
        d = np.array([float(r[0]) for r in rows])
        I = np.array([float(r[1]) for r in rows])
        s = TAG_WL / (2.0 * d)
        ok = s < 1
        t = np.full(len(d), np.nan)
        t[ok] = 2.0 * np.degrees(np.arcsin(s[ok]))
        top = np.argsort(-I)[:3]
        e = dict(name=name, src=src, rid=rid, note=note, el=set(els.split()), d=d, I=I, t=t,
                 hkl=[r[2] if len(r) > 2 else "" for r in rows], top3=d[top], key=src + ":" + rid, nrec=n_r + n_a, n_r=n_r, n_a=n_a,
                 rir=rir)
        by_name.setdefault(name.lower(), []).append(len(ents))
        by_base.setdefault(base_name(name), []).append(len(ents))
        ents.append(e)
    MINDB = dict(e=ents, by_name=by_name, by_base=by_base, by_key={e["key"]: e for e in ents},
                 version=raw.get("version", ""), cite=raw.get("cite", []), rir=raw.get("rir", {}),
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

    last = {}

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
        # 每個礦物 100 % 峰（無方位效應時）的積分強度 = 非軸向那一列的係數
        last["scales"] = [float(a[(own == i) & (kind != 2)][0]) if ((own == i) & (kind != 2)).any() else 0.0
                          for i in range(len(ks))]
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
    scales = list(last.get("scales", [0.0] * len(sel)))
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
        # 100 % 峰的積分強度換成 2θ 單位（q 網格 → 2θ：除以 cos θ）
        e = ents[k]
        j100 = int(np.argmax(e["I"]))
        sn = L1 / (2 * e["d"][j100] * (1 + par[k][1]))
        cos_t = math.cos(math.asin(min(sn, 1.0))) if sn < 1 else 1.0
        out.append(dict(name=ents[k]["name"], key=ents[k]["key"], tier=tiers.get(k, "possible"),
                        scale=max(0.0, scales[i]) / cos_t,
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

# ── 全岩半定量：RIR 法（Chung 1974；Hubbard & Snyder 1988）──────────────────────────────
#   w_i ∝ I_i / RIR_i，I_i = 擬合得到的該礦物 100 % 峰積分強度；RIR = I/I剛玉（AMCSD 由晶體結構計算）
CLAY_LIKE = set("""illite halloysite kaolinite dickite nacrite montmorillonite nontronite beidellite saponite hectorite
vermiculite palygorskite sepiolite glauconite celadonite chlorite clinochlore chamosite muscovite phengite paragonite
biotite phlogopite annite talc pyrophyllite antigorite lizardite chrysotile""".split())
INTERNAL_STD = {"corundum": ("剛玉", {"Al"}), "zincite": ("氧化鋅（紅鋅礦）", {"Zn"})}
ID_LAST = {}


def _inst_width(x, y):
    """由孤立的強峰量測半高寬隨角度的變化：FWHM(2θ) = a + b·tanθ（實驗室與同步輻射的峰都會隨角度變寬）"""
    obs = detect_peaks(x, y)
    step = float(np.median(np.diff(x))) or 0.02
    yb = y - snip_background(y, max(1, min(int(round(1.5 / step / 2)), len(y) // 2 - 1)))
    ts = obs["t"][np.argsort(-obs["I"])][:30]
    pts = []
    for t in ts:
        i = int(np.clip(np.searchsorted(x, t), 1, len(x) - 2))
        h = yb[i]
        if h <= 0:
            continue
        a = i
        while a > 0 and yb[a] > h / 2 and x[i] - x[a] < 1.5:
            a -= 1
        b = i
        while b < len(x) - 1 and yb[b] > h / 2 and x[b] - x[i] < 1.5:
            b += 1
        fw = x[b] - x[a]
        if fw <= 2 * step or fw >= 1.5:
            continue
        if np.any((np.abs(obs["t"] - t) > 1e-6) & (np.abs(obs["t"] - t) < 2.0 * fw)):
            continue                                   # 旁邊有其他峰：不夠孤立
        pts.append((t, fw))
    if len(pts) < 4:
        w = float(np.median([p[1] for p in pts])) if pts else 0.1
        return lambda t: np.full(np.shape(t), w, float)
    T = np.tan(np.radians(np.array([p[0] for p in pts]) / 2))
    W = np.array([p[1] for p in pts])
    keep = np.ones(len(T), bool)
    for _ in range(2):
        A = np.stack([np.ones(keep.sum()), T[keep]], 1)
        c = np.linalg.lstsq(A, W[keep], rcond=None)[0]
        r = W - (c[0] + c[1] * T)
        mad = 1.4826 * float(np.median(np.abs(r[keep]))) + 1e-9
        keep = np.abs(r) <= 2.5 * mad
        if keep.sum() < 3:
            break
    a0, b0 = float(c[0]), float(max(c[1], 0.0))
    if a0 <= 0:
        a0, b0 = float(np.median(W)), 0.0
    return lambda t: a0 + b0 * np.tan(np.radians(np.asarray(t, float) / 2))


def quant_refit(x, y, kalpha2, phases):
    """
    定量用的重新擬合：已鑑定的礦物以 2θ 空間的峰形重建（半高寬隨角度變化），並逐一精修每個礦物的
    2θ 位移、晶格縮放與峰寬，回傳每個礦物 100 % 峰（Kα1）的積分強度。
    鑑定階段為了速度使用固定峰寬與較粗的晶格格點，高角度峰的位置與強度會有偏差，不適合直接定量。
    """
    o = np.argsort(x)
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    step = float(np.median(np.diff(x))) or 0.02
    bg = snip_background(y, max(2, int(round(3.0 / step / 2))))
    yb = y - bg
    winst = _inst_width(x, y)
    lo, hi = float(x[0]), float(x[-1])
    u = 2 * (x - lo) / (hi - lo) - 1
    B = np.stack([np.polynomial.legendre.legval(u, np.eye(6)[k]) for k in range(6)], 1)
    floor = max(1e-9, 0.01 * float(np.median(np.abs(y))))
    w = np.maximum(y, floor) ** -0.15
    w /= w.mean()
    Bw, ybw = B * w[:, None], yb * w
    db = mindb()
    ents = [db["by_key"][p["key"]] for p in phases]
    pos = [fp_po_family(e, lo, hi) for e in ents]
    n = len(phases)
    cache = {}

    def prof(i, shift, eps, mult, eta, axial):
        key = (i, round(shift, 4), round(eps, 5), mult, eta, axial)
        if key in cache:
            return cache[key]
        e = ents[i]
        m = np.isfinite(e["t"]) & (e["I"] >= 1)
        if axial:
            m &= pos[i]
        d = e["d"][m] * (1 + eps)
        I = e["I"][m] / 100.0
        P = np.zeros(len(x))
        for lam, f in ((L1, 1.0), (L2, 0.5)) if kalpha2 else ((L1, 1.0),):
            sn = lam / (2 * d)
            ok = sn < 1
            t = 2 * np.degrees(np.arcsin(sn[ok])) + shift
            for tt, ii in zip(t, I[ok]):
                if tt < lo - 2 or tt > hi + 2:
                    continue
                fw = mult * float(winst(tt))
                a, b = np.searchsorted(x, [tt - 15 * fw, tt + 15 * fw])
                if b > a:
                    P[a:b] += f * ii * _pv_unit(x[a:b], tt, fw, eta)
        if len(cache) > 4000:
            cache.clear()
        cache[key] = P
        return P

    def solve(par, eta):
        """par[i] = [shift, eps, mult]；回傳 (加權殘差平方和, 各礦物 100 % 峰強度)"""
        def build(fixed):
            cols, own, kind = [], [], []
            for i in range(n):
                sh, ep, mu = par[i]
                pf = prof(i, sh, ep, mu, eta, False)
                if pos[i] is None:
                    cols.append(pf); own.append(i); kind.append(0)
                else:
                    pb = prof(i, sh, ep, mu, eta, True)
                    if i in fixed:
                        cols.append(pf - pb + fixed[i] * pb); own.append(i); kind.append(0)
                    else:
                        cols += [pf - pb, pb]; own += [i, i]; kind += [1, 2]
            return np.stack(cols, 1), np.array(own), np.array(kind)
        fixed = {}
        for _ in range(2):
            D, own, kind = build(fixed)
            k = D.shape[1]
            A = np.hstack([D * w[:, None], Bw])
            th = _lsq_peaks_bg(A, ybw, k)
            new = {}
            for i in range(n):
                r_i = np.where((own == i) & (kind == 1))[0]
                p_i = np.where((own == i) & (kind == 2))[0]
                if len(r_i) and len(p_i):
                    ar, ap = th[r_i[0]], th[p_i[0]]
                    if ap > 4.0 * ar:
                        new[i] = 4.0          # 方位效應最多放大或減弱 4 倍
                    elif ap < ar / 4.0:
                        new[i] = 0.25
            if not new or new == fixed:
                break
            fixed = {**fixed, **new}
        r = ybw - A @ th
        return float(r @ r), [float(th[np.where((own == i) & (kind != 2))[0][0]]) for i in range(n)]

    par = [[float(p.get("shift", 0.0)), float(p.get("eps", 0.0)), 1.0] for p in phases]
    eta = 0.5
    best = solve(par, eta)

    def try_set(i, j, vals):
        nonlocal best, par
        for v in vals:
            tr = [list(q) for q in par]
            tr[i][j] = v
            ss = solve(tr, eta)
            if ss[0] < best[0]:
                best, par = ss, tr

    for _pass in range(2):
        for e2 in (0.2, 0.5, 0.8):
            ss = solve(par, e2)
            if ss[0] < best[0]:
                best, eta = ss, e2
        for i in range(n):
            s0, e0, _m = par[i]
            try_set(i, 0, [s0 + d for d in (-0.06, -0.04, -0.02, 0.02, 0.04, 0.06)])
            s0, e0, _m = par[i]
            try_set(i, 1, [e0 + d for d in (-0.003, -0.002, -0.001, -0.0005, 0.0005, 0.001, 0.002, 0.003)])
            try_set(i, 2, [m for m in (0.6, 0.8, 1.0, 1.25, 1.6, 2.2, 3.0) if m != par[i][2]])
    return [max(0.0, s) for s in best[1]]


def rir_of(e):
    """
    (RIR, 下限, 上限, 結構數, 來源)。比對到的 AMCSD 記錄有自己的 RIR 時優先使用（和擬合用的相對強度一致）；
    RRUFF 記錄沒有 RIR，改用同名礦物所有 AMCSD 結構的中位數。範圍取同名礦物的 10–90 百分位。
    """
    st = mindb().get("rir") or {}
    b = base_name(e["name"])
    for key in (e["name"].lower(), b, b.split("-")[0], b.split()[0] if b.split() else b):
        v = st.get(key)
        if v:
            med, p10, p90, n = v
            val = e.get("rir") or med
            return val, val * p10 / med, val * p90 / med, n, "entry" if e.get("rir") else "mineral"
    if e.get("rir"):
        return e["rir"], e["rir"], e["rir"], 1, "entry"
    g = fp_group_of(e)
    if g:
        vals = sorted(st[m][0] for m in MINERAL_GROUPS[g].split() if m in st)
        if vals:
            return vals[len(vals) // 2], vals[0], vals[-1], len(vals), "group"
    return None


def is_clay_like(e):
    b = base_name(e["name"]).split("-")[0].split()[0] if e["name"].strip() else ""
    g = fp_group_of(e) or ""
    return b in CLAY_LIKE or any(k in g for k in ("kaolin", "smectite", "chlorite", "mica", "serpentine"))


def semi_quant(phases, std_key=None, std_wt=None, seed=0):
    """phases: [{name, key, scale, tier}] → 各礦物 wt%（中位數與 16–84 % 範圍），可選內標求非晶質含量"""
    db = mindb()
    rows = []
    for p in phases:
        e = db["by_key"][p["key"]]
        rows.append(dict(name=p["name"], key=p["key"], src=e["src"], rid=e["rid"], tier=p.get("tier"),
                         scale=float(p.get("scale") or 0.0), rir=rir_of(e), clay=is_clay_like(e)))
    warn = []
    std_i = None
    if std_key:
        std_i = next((i for i, r in enumerate(rows) if base_name(r["name"]).split("-")[0] == std_key), None)
        if std_i is None:
            warn.append("辨識結果中沒有找到內標（%s），改用「所有礦物合計 100 %%」計算。" % INTERNAL_STD[std_key][0])
        elif not std_wt or not (0 < std_wt < 100):
            warn.append("請輸入內標的加入量（wt%），目前改用「所有礦物合計 100 %」計算。")
            std_i = None
    use = [i for i, r in enumerate(rows) if r["rir"] and r["scale"] > 0]
    if not use:
        return dict(ok=False, why="沒有可計算的礦物（缺少 RIR 或強度）。", rows=[], warn=warn)
    rng = np.random.default_rng(seed)
    N = 4000
    R0 = np.ones(len(rows))
    R = np.ones((len(rows), N))
    for i in use:
        med, lo, hi, n, _src = rows[i]["rir"]
        if std_i is not None and i == std_i and std_key == "corundum":
            med = lo = hi = 1.0                     # 依定義
        elif n < 3 or hi <= lo * 1.02:
            lo, hi = med / 1.2, med * 1.2           # 結構太少時假設 ±20 %
        R0[i] = med
        R[i] = np.exp(rng.uniform(math.log(lo), math.log(hi), N)) if hi > lo else med
    sc = np.array([r["scale"] for r in rows])
    mask = np.zeros(len(rows), bool)
    mask[use] = True
    # 擬合強度的不確定度：以已知配比的模擬混合物測試，各礦物強度誤差約 ±5–7 %
    Ifac = np.exp(0.07 * rng.standard_normal((len(rows), N)))

    def wt(Rm):
        if Rm.ndim == 2:
            X = np.where(mask[:, None], sc[:, None] * Ifac / Rm, 0.0)
        else:
            X = np.where(mask, sc / Rm, 0.0)
        if std_i is None:
            return 100.0 * X / X.sum(0)
        Wmix = std_wt * X / X[std_i]                # 混合物（含內標）中的 wt%
        return Wmix / (1.0 - std_wt / 100.0)        # 換回原樣品
    w0, wm = wt(R0), wt(R)
    out = []
    for i, r in enumerate(rows):
        if i == std_i:
            continue
        d = dict(name=r["name"], key=r["key"], src=r["src"], rid=r["rid"], tier=r["tier"], clay=r["clay"],
                 rir=None if not r["rir"] else round(R0[i], 3), rir_lo=None if not r["rir"] else r["rir"][1],
                 rir_hi=None if not r["rir"] else r["rir"][2], rir_n=0 if not r["rir"] else r["rir"][3],
                 rir_src=None if not r["rir"] else r["rir"][4])
        if mask[i]:
            v0 = float(w0[i])
            d.update(wt=v0, lo=min(v0, float(np.percentile(wm[i], 16))), hi=max(v0, float(np.percentile(wm[i], 84))))
        else:
            d.update(wt=None, lo=None, hi=None)
        out.append(d)
    amorph = None
    if std_i is not None:
        keep = [i for i in use if i != std_i]
        a0 = 100.0 - float(w0[keep].sum())
        am = 100.0 - wm[keep].sum(0)
        amorph = dict(wt=max(0.0, a0), lo=max(0.0, min(a0, float(np.percentile(am, 16)))),
                      hi=max(0.0, a0, float(np.percentile(am, 84))))
        if a0 < -5:
            warn.append("結晶相合計超過 100 %%（%.0f %%），非晶質以 0 計；可能是 RIR 誤差或內標加入量不正確。" % (100 - a0))
    if any(r["clay"] for r in out if r["wt"] is not None):
        warn.append("黏土礦物與雲母的 RIR 受優選方位與層間無序影響，wt% 誤差可能很大；黏土的比例建議以定向片分析或 Rietveld 為準。")
    if std_i is None:
        warn.append("沒有內標：假設所有礦物都是結晶相且都已鑑定（合計 100 %）。若含非晶質或未鑑定的礦物，各礦物的 wt% 都會偏高。")
    miss = [r["name"] for r in out if r["wt"] is None]
    if miss:
        warn.append("以下礦物沒有 RIR，未計入：" + "、".join(miss))
    out.sort(key=lambda r: -(r["wt"] or -1))
    return dict(ok=True, mode="internal" if std_i is not None else "sum100", std=std_key if std_i is not None else None,
                std_wt=std_wt if std_i is not None else None, rows=out, amorph=amorph, warn=warn)


RIR_REFS = {
    "chung": "Chung, F.H., 1974. Quantitative interpretation of X-ray diffraction patterns of mixtures. I. Matrix-flushing method for quantitative multicomponent analysis. Journal of Applied Crystallography 7, 519–525.",
    "hubbard": "Hubbard, C.R., Snyder, R.L., 1988. RIR – measurement and use in quantitative XRD. Powder Diffraction 3, 74–77.",
    "rruff": "Lafuente, B., Downs, R.T., Yang, H., Stone, N., 2015. The power of databases: the RRUFF project. In: Armbruster, T., Danisi, R.M. (Eds.), Highlights in Mineralogical Crystallography. De Gruyter, Berlin, pp. 1–30.",
    "amcsd": "Downs, R.T., Hall-Wallace, M., 2003. The American Mineralogist Crystal Structure Database. American Mineralogist 88, 247–250.",
}


def quant_methods(q, sync_wl=None):
    p = ["Mineral phases were identified by whole-pattern fitting of calculated reference patterns from the RRUFF and "
         "AMCSD databases (Lafuente et al., 2015; Downs and Hall-Wallace, 2003)."]
    if sync_wl:
        p.append("Synchrotron patterns (λ = %s Å) were converted to the 2θ scale of Cu Kα1 radiation (λ = 1.540598 Å) "
                 "before analysis." % wl(sync_wl))
    p.append("Semi-quantitative phase abundances were estimated with the reference intensity ratio (RIR) method "
             "(Chung, 1974; Hubbard and Snyder, 1988) from the fitted integrated intensity of the strongest reflection "
             "of each phase. RIR values (I/I_corundum) were taken as the median of the values calculated for all AMCSD "
             "structures of the same mineral.")
    if q.get("mode") == "internal":
        std = {"corundum": "corundum (α-Al2O3)", "zincite": "zincite (ZnO)"}[q["std"]]
        p.append("A known amount of %s (%s wt%%) was added as an internal standard, which allows the amorphous and "
                 "unidentified fraction to be estimated as the difference from 100 wt%%." % (std, ("%g" % q["std_wt"])))
    else:
        p.append("The crystalline phases were assumed to sum to 100 wt%.")
    p.append("Phase intensities were obtained by refitting the identified phases with angle-dependent peak widths and "
             "refined lattice parameters. Reported ranges (16th–84th percentiles) combine the spread of RIR values among "
             "structures of the same mineral with an intensity-fitting uncertainty of ±7%, and do not include "
             "microabsorption, preferred orientation or particle-statistics effects; values for clay minerals and micas "
             "are particularly uncertain.")
    p.append("Data processing was carried out with XRD Studio v%s." % APP_VERSION)
    return " ".join(p) + "\n\nReferences\n" + "\n".join(RIR_REFS[k] for k in ("chung", "amcsd", "hubbard", "rruff"))


def quant_export(S, a):
    q = ID_LAST.get("quant")
    if not q or not q.get("ok"):
        raise UserError("請先執行「自動全譜辨識」。")
    o = io.StringIO()
    w = csv.writer(o)
    w.writerow(["phase", "database", "id", "tier", "wt_%", "wt_16pct", "wt_84pct", "RIR", "RIR_p10", "RIR_p90", "RIR_n_structures",
                "clay_or_mica"])
    for r in q["rows"]:
        w.writerow([r["name"], {"R": "RRUFF", "A": "AMCSD"}.get(r["src"], r["src"]), r["rid"], r["tier"],
                    _f(r["wt"], 1), _f(r["lo"], 1), _f(r["hi"], 1), _f(r["rir"], 3), _f(r["rir_lo"], 3), _f(r["rir_hi"], 3),
                    r["rir_n"], "yes" if r["clay"] else ""])
    if q.get("amorph"):
        am = q["amorph"]
        w.writerow(["amorphous + unidentified", "", "", "", _f(am["wt"], 1), _f(am["lo"], 1), _f(am["hi"], 1)])
    files = [("semi_quantitative.csv", "﻿" + o.getvalue()), ("methods.txt", quant_methods(q, ID_LAST.get("sync_wl")))]
    if ID_LAST.get("png"):
        files.append(("fit.png", ID_LAST["png"]))
    nm = safe_name(ID_LAST.get("name") or "sample", "sample")
    return {"name": nm + "_semi_quant.zip", "mime": "application/zip"}, make_zip(files)


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
    std_key = a.get("std") if a.get("std") in INTERNAL_STD else None
    if std_key and allowed:
        allowed |= INTERNAL_STD[std_key][1]          # 內標的元素一定要能被搜尋到
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
    qp = res["phases"]
    if qp:
        sc = quant_refit(x, y, kalpha2, qp)
        qp = [dict(p, scale=s) for p, s in zip(qp, sc)]
    quant = semi_quant(qp, std_key, fnum(a.get("std_wt")))
    ID_LAST.clear()
    ID_LAST.update(quant=quant, png=png, name=d0.get("base") or d0["name"],
                   sync_wl=eff["src"] if eff["type"] != "inhouse" and eff["convert"] else None)
    if quant.get("ok"):
        quant["methods"] = quant_methods(quant, ID_LAST["sync_wl"])
    return dict(phases=phases, rwp=round(res["rwp"], 4), pool=res["pool"], kalpha2=kalpha2, quant=quant,
                secs=round(_time.time() - t0, 1)), png


# ═════════════════════════════════════════════════════════════════════════════
# 14. 定量（Rietveld）：匯出給 Profex / GSAS-II 的檔案，以及把精修結果畫成期刊格式的 Rietveld 圖
# ═════════════════════════════════════════════════════════════════════════════
RV = {}   # 目前載入的精修結果


def _amcsd_for(name):
    """同名礦物在資料庫中的 AMCSD 記錄（Rietveld 需要晶體結構，RRUFF 峰表沒有原子座標）"""
    es = [e for e in lookup(name) if e["src"] == "A"]
    return es[0] if es else None


def rietveld_package(S, items, a):
    iid = a.get("id")
    if iid not in DATA:
        raise UserError("請先選擇要精修的曲線。")
    d0 = DATA[iid]
    eff = effective(d0, S)
    # 原始波長的數據（不做換算）：同步輻射用該次 beamtime 的波長；已換算的檔案只能以 Cu Kα1 輸出
    lam = eff["src"]
    x, y = d0["x"].astype(float), d0["y"].astype(float)
    esd = None
    if d0["rows"] and len(d0["rows"][0]) >= 3:
        try:
            esd = np.array([float(r[2]) for r in d0["rows"]])
        except (ValueError, IndexError):
            esd = None
    if esd is None or not np.all(np.isfinite(esd)) or np.all(esd <= 0):
        esd = np.sqrt(np.maximum(y, 1.0))
    base = safe_name(d0.get("base") or os.path.splitext(d0["name"])[0], "data")
    conv_note = ""
    if eff["type"] == "sync_conv":
        conv_note = "（這個檔案已換算成 Cu Kα1 的 2θ；強度沒有換算，精修時請以 λ = 1.540598 Å、無 Kα2 設定）"
    xye = io.StringIO()
    xye.write("/* %s — wavelength %s A — exported by XRD Studio v%s */\n" % (d0["name"], wl(lam), APP_VERSION))
    for xi, yi, ei in zip(x, y, esd):
        xye.write("%.5f %.4f %.4f\n" % (xi, yi, ei))
    names = [str(n).strip() for n in (a.get("phases") or []) if str(n).strip()]
    q = ID_LAST.get("quant") or {}
    qwt = {r["name"]: r for r in q.get("rows", [])} if q.get("ok") else {}
    rows = io.StringIO()
    w = csv.writer(rows)
    w.writerow(["phase", "AMCSD_code_for_CIF", "approx_wt%_RIR", "note"])
    for n in names:
        e = _amcsd_for(n)
        r = qwt.get(n)
        w.writerow([n, e["rid"] if e else "", _f(r["wt"], 1) if r and r.get("wt") is not None else "",
                    ("clay/mica: use a disordered structure model if available" if r and r.get("clay") else "")])
    src_txt = ("同步輻射，波長 %s Å（請確認與該次 beamtime 相同）" % wl(lam)) if eff["type"] == "sync_raw" else \
              ("實驗室 Cu Kα（Kα1 = 1.540598、Kα2 = 1.544426 Å）" if eff["type"] == "inhouse" else "已換算為 Cu Kα1")
    readme = """XRD Studio → Rietveld 精修（Profex / GSAS-II）
=========================================

數據：%(base)s.xye
  光源：%(src)s%(conv)s
  欄位：2θ、強度、誤差（原始檔沒有誤差欄時以 √強度 代替）。2θ 為原始波長，沒有換算。

礦物：phases.csv（自動全譜辨識的結果；approx_wt%%_RIR 只是 RIR 半定量的估計，可當起始值參考）

建議流程（Profex，免費：https://www.profex-xrd.org）
1. 儀器設定：
   - 實驗室繞射儀：選擇和你的儀器相符的設定檔（狹縫、測角儀半徑、是否有單色器）。
   - 同步輻射：用「同一次 beamtime、同樣架設」量的 LaB6 或 CeO2 標準品圖譜建立儀器峰形，
     並設定波長 %(wl)s Å、偏振（同步輻射為線偏振）。沒有標準品就無法正確描述儀器峰形。
2. File → Open 開啟 %(base)s.xye。
3. 加入礦物：優先使用 Profex 內建的結構檔（黏土礦物有專用的無序結構模型）；
   沒有的礦物，到 AMCSD（https://rruff.geo.arizona.edu/AMS/amcsd.php）用 phases.csv 的編號下載 CIF，
   再用 Profex 的「Import CIF」轉成結構檔。
4. 精修後匯出結果：
   - Profex：精修結果的 .dia 檔（與 .par 同資料夾）即可直接載入 XRD Studio 畫 Rietveld 圖。
   - GSAS-II：Export → Powder data as → CSV file（再加上 Reflection list as CSV 可得精確的峰位刻度）。
5. 論文中請寫明：精修軟體與版本、各礦物使用的結構來源（AMCSD 編號或 Profex 結構檔名稱）、Rwp 與 GOF、
   是否使用內標，以及 wt%% 的誤差（精修給的 ESD 通常低估真實誤差）。
""" % dict(base=base, src=src_txt, conv=conv_note, wl=wl(lam))
    files = [(base + ".xye", xye.getvalue()), ("phases.csv", "﻿" + rows.getvalue()), ("README_Rietveld.txt", readme)]
    return {"name": base + "_for_Rietveld.zip", "mime": "application/zip"}, make_zip(files)


_RV_NAMES = {"obs": ("y_obs", "yobs", "obs", "iobs", "observed", "intensity", "i_obs"),
             "calc": ("y_calc", "ycalc", "calc", "icalc", "calculated", "i_calc", "ycal"),
             "bkg": ("y_bkg", "ybkg", "bkg", "background", "ibkg", "back", "backg", "i_bkg"),
             "skip": ("weight", "obs-calc", "diff", "difference", "esd", "sigma", "y_diff", "yobs-ycal")}


def _rv_tokens(s):
    s = s.strip()
    parts = s.split(",") if "," in s else re.split(r"[\t;]|\s+", s)
    return [t.strip().strip('"').strip("'").strip() for t in parts if t.strip().strip('"').strip("'").strip()]


def read_refined(name, text):
    """讀精修結果：GSAS-II 的粉末 CSV / 反射列表 CSV、Profex(BGMN) .dia，或任何「2θ、觀測、計算[、背景、各相]」的欄位檔"""
    lines = text.splitlines()
    low = text.lower()
    if '"h","k","l"' in low.replace(" ", "") or re.search(r"^\s*h\s*,\s*k\s*,\s*l\s*,", low, re.M):
        # GSAS-II 反射列表
        phases, refl, cols = {}, {}, None
        for ln in lines:
            t = _rv_tokens(ln)
            if not t:
                continue
            tl = [v.lower() for v in t]
            if tl[:3] == ["h", "k", "l"]:
                cols = tl
                continue
            if cols is None:
                if len(t) == 2 and _isnum(t[1]) and not _isnum(t[0]):
                    phases[int(float(t[1]))] = t[0]
                continue
            if len(t) == len(cols) and all(_isnum(v) for v in t):
                pos_i = next((i for i, c in enumerate(cols) if c in ("2-theta", "2theta", "2θ", "pos")), 3)
                ph = int(float(t[cols.index("phase #")])) if "phase #" in cols else 0
                refl.setdefault(ph, []).append(float(t[pos_i]))
        if not refl:
            raise UserError("%s：看起來是反射列表，但讀不到峰位。" % name)
        return dict(kind="refl", refl={phases.get(k, "Phase %d" % (k + 1)): sorted(v) for k, v in refl.items()})
    header, rows = None, []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith(("//", "#", "/*")) and not rows:
            if s.startswith(("//", "#")) and not rows:
                hdr = _rv_tokens(s.lstrip("/#").strip())
                if len(hdr) >= 3:
                    header = hdr
            continue
        t = _rv_tokens(s)
        if len(t) >= 3 and _isnum(t[0]) and _isnum(t[1]):
            rows.append([float(v) if _isnum(v) else float("nan") for v in t])
        elif not rows and len(t) >= 3:
            header = t
    if len(rows) < 20:
        raise UserError("%s：讀不到精修結果的數據欄（至少要有 2θ、觀測、計算三欄）。" % name)
    from collections import Counter
    ncol = Counter(len(r) for r in rows).most_common(1)[0][0]
    arr = np.array([r for r in rows if len(r) == ncol], float)
    names = [h.lower() for h in header] if header and len(header) == ncol else None
    roles = {}
    if names:
        for i, nm in enumerate(names[1:], 1):
            for role in ("obs", "calc", "bkg"):
                if nm in _RV_NAMES[role] and role not in roles:
                    roles[role] = i
                    break
    if "obs" not in roles or "calc" not in roles:
        roles = {"obs": 1, "calc": 2}
        if ncol >= 4:
            roles["bkg"] = 3
    used = set(roles.values()) | {i for i, nm in enumerate(names or []) if nm in _RV_NAMES["skip"]}
    extra = [i for i in range(1, ncol) if i not in used]
    cols = [dict(i=i, name=(header[i] if names else "第 %d 欄" % (i + 1))) for i in range(ncol)]
    return dict(kind="pattern", name=name, arr=arr, roles=roles, extra=extra, cols=cols)


def _rv_ticks(R, phases, lam):
    """各礦物的峰位刻度：有反射列表就用它；否則用資料庫峰表，並把晶格微調到和計算圖譜的峰對齊"""
    arr, ro = R["arr"], R["roles"]
    x = arr[:, 0]
    calc = arr[:, ro["calc"]] - (arr[:, ro["bkg"]] if "bkg" in ro else np.min(arr[:, ro["calc"]]))
    calc = np.clip(calc, 0, None)
    refl = RV.get("refl") or {}
    out = []
    for ph in phases:
        nm = ph["name"]
        hit = next((v for k, v in refl.items() if base_name(k) == base_name(nm) or k.lower() == nm.lower()), None)
        if hit:
            pos = np.array([t for t in hit if x[0] <= t <= x[-1]])
            out.append(dict(ph, pos=pos, src="refl"))
            continue
        es = lookup(nm)
        if not es:
            out.append(dict(ph, pos=np.array([]), src="none"))
            continue
        e = next((e for e in es if e["src"] == "A"), es[0])
        m = e["I"] >= 2
        d, I = e["d"][m], e["I"][m]
        best = (-1.0, 0.0)
        for eps in np.arange(-0.015, 0.01501, 0.0005):
            sn = lam / (2 * d * (1 + eps))
            ok = sn < 1
            t = 2 * np.degrees(np.arcsin(sn[ok]))
            v = float((np.interp(t, x, calc, left=0, right=0) * I[ok]).sum())
            if v > best[0]:
                best = (v, eps)
        sn = lam / (2 * d * (1 + best[1]))
        ok = sn < 1
        t = 2 * np.degrees(np.arcsin(sn[ok]))
        out.append(dict(ph, pos=np.sort(t[(t >= x[0]) & (t <= x[-1])]), src="db"))
    return out


def rietveld_figure(S, opts, fmt="png", dpi=300, preview=False):
    R = RV.get("pattern")
    if not R:
        raise UserError("請先載入精修結果。")
    S = merge_settings(S or {})
    arr, ro = R["arr"], dict(R["roles"])
    for k in ("obs", "calc", "bkg"):
        v = opts.get("col_" + k)
        if v is not None and v != "":
            if int(v) < 0:
                ro.pop(k, None)
            else:
                ro[k] = int(v)
    if "obs" not in ro or "calc" not in ro:
        raise UserError("請指定觀測與計算強度的欄位。")
    R2 = dict(R, roles=ro)
    x, yo, yc = arr[:, 0], arr[:, ro["obs"]], arr[:, ro["calc"]]
    yb = arr[:, ro["bkg"]] if "bkg" in ro else None
    x0 = fnum(opts.get("xmin"), float(x.min()))
    x1 = fnum(opts.get("xmax"), float(x.max()))
    m = (x >= x0) & (x <= x1)
    x, yo, yc = x[m], yo[m], yc[m]
    yb = yb[m] if yb is not None else None
    lam = fnum(opts.get("wl"), TAG_WL)
    phases = [p for p in (opts.get("phases") or []) if p.get("name")]
    ticks_all = _rv_ticks(dict(R2, arr=arr[m]), phases, lam)
    ticks = [t for t in ticks_all if t["src"] != "none"]
    diff = yo - yc
    top, bot = float(max(yo.max(), yc.max())), float(min(yo.min(), yc.min()))
    rng_ = top - bot or 1.0
    nt = len(ticks)
    gap = 0.045 * rng_
    tick_top = bot - 0.05 * rng_
    tick_bot = tick_top - nt * gap
    dmax = float(np.percentile(np.abs(diff), 99.8)) or 1.0
    d0 = tick_bot - 0.04 * rng_ - dmax
    W = S["width_mm"] / 25.4
    H = S["height_mm"] / 25.4
    with matplotlib.rc_context(make_rc(S)):
        fig = Figure(figsize=(W, H), dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        lw = S["lw"]
        ms = max(1.0, 2.2 * lw)
        step = max(1, len(x) // 2500)
        ax.plot(x[::step], yo[::step], ls="none", marker="o", ms=ms, mfc="none", mec="black", mew=0.35 * lw / 0.75,
                label="Observed", zorder=2)
        ax.plot(x, yc, color="#D62728", lw=lw, label="Calculated", zorder=3)
        if yb is not None and opts.get("show_bkg", True):
            ax.plot(x, yb, color="#8e8e93", lw=lw * 0.7, label="Background", zorder=1)
        ax.plot(x, diff + d0, color="#1F5FAD", lw=lw * 0.8, label="Difference", zorder=3)
        for i, tk in enumerate(ticks):
            yy = tick_top - (i + 0.5) * gap
            c = TICK_COLORS[i % len(TICK_COLORS)]
            if len(tk["pos"]):
                ax.vlines(tk["pos"], yy - 0.36 * gap, yy + 0.36 * gap, color=c, lw=lw * 0.6, zorder=2)
            lab = tk["name"]
            if tk.get("wt") not in (None, ""):
                lab += "  %s wt%%" % tk["wt"] + (" (± %s)" % tk["err"] if tk.get("err") not in (None, "") else "")
            ax.text(1.005, yy, lab, transform=blended_transform_factory(ax.transAxes, ax.transData), ha="left",
                    va="center", fontsize=S["fs_legend"], color=c, clip_on=False)
        ax.set_xlim(x0, x1)
        ax.set_ylim(d0 - dmax * 1.3, top + 0.06 * rng_)
        ax.set_yticks([])
        ax.set_xlabel(fmt_label(opts.get("xlabel") or "2θ (°)", S), fontsize=S["fs_label"])
        ax.set_ylabel("Intensity (a.u.)", fontsize=S["fs_label"])
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        stats = []
        if opts.get("rwp") not in (None, ""):
            stats.append("$R_{wp}$ = %s%%" % opts["rwp"])
        if opts.get("gof") not in (None, ""):
            stats.append("GOF = %s" % opts["gof"])
        if opts.get("show_wl", True):
            stats.append("λ = %s Å" % wl(lam))
        if stats:
            ax.text(0.015, 0.97, "\n".join(stats), transform=ax.transAxes, ha="left", va="top", fontsize=S["fs_legend"])
        ax.legend(loc="upper right", frameon=False, fontsize=S["fs_legend"], handlelength=1.4, markerscale=1.6)
        buf = io.BytesIO()
        kw = dict(format=fmt, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.03)
        try:
            fig.savefig(buf, **(dict(kw, pil_kwargs={"compression": "tiff_lzw"}) if fmt == "tiff" else kw))
        except Exception:      # 舊版 Pillow 不支援 LZW
            buf = io.BytesIO()
            fig.savefig(buf, **kw)
    miss = [t["name"] for t in ticks_all if t["src"] == "none"]
    return buf.getvalue(), dict(ticks=[dict(name=t["name"], n=int(len(t["pos"])), src=t["src"]) for t in ticks_all], missing=miss)


def rv_load(a):
    out = []
    for f in a.get("files") or []:
        r = read_refined(str(f.get("name") or "result"), str(f.get("text") or ""))
        if r["kind"] == "refl":
            RV["refl"] = r["refl"]
            out.append(dict(name=f.get("name"), kind="refl", phases=list(r["refl"].keys())))
        else:
            RV["pattern"] = r
            out.append(dict(name=f.get("name"), kind="pattern", cols=r["cols"], roles=r["roles"], extra=r["extra"],
                            n=int(len(r["arr"])), xr=[float(r["arr"][:, 0].min()), float(r["arr"][:, 0].max())]))
    return dict(loaded=out, has_pattern="pattern" in RV, refl_phases=list((RV.get("refl") or {}).keys()))


# ═════════════════════════════════════════════════════════════════════════════
# 13. 黏土礦物（定向片：AD 風乾 / EG 乙二醇 / 加熱 / 甲醯胺）
#     底面反射峰擬合 → 依處理前後的 d 值變化判讀 → Biscaye (1965) 半定量 → Kübler / Esquevin 指標
# ═════════════════════════════════════════════════════════════════════════════
CLAY_ORDER = ["AD", "EG", "H350", "H550", "FA"]
CLAY_TREAT_ZH = {"AD": "AD 風乾", "EG": "EG 乙二醇", "H350": "加熱 350 °C", "H550": "加熱 550 °C", "FA": "甲醯胺 FA"}
CLAY_TREAT_FIG = {"AD": "AD", "EG": "EG", "H350": "350 °C", "H550": "550 °C", "FA": "FA"}
CLAY_TREAT_EN = {"AD": "air-dried (AD)", "EG": "after ethylene-glycol solvation (EG)", "H350": "after heating to 350 °C",
                 "H550": "after heating to 550 °C", "FA": "after formamide treatment"}
CLAY_LAST = {}   # 最近一次分析結果（下載圖檔、CSV 用）


def _t_of_d(d):
    return 2.0 * math.degrees(math.asin(min(1.0, TAG_WL / (2.0 * d))))


def _d_of_t(t):
    return TAG_WL / (2.0 * math.sin(math.radians(t / 2.0)))


def _cc(key, dlo, dhi, wlo, whi, d0, w0, tie=None):
    """一個要擬合的峰：d 值範圍（Å）、半高寬範圍（°2θ, Cu Kα）、初始值；tie = 與另一個峰共用半高寬"""
    return dict(key=key, dlo=dlo, dhi=dhi, wlo=wlo, whi=whi, d0=d0, w0=w0, tie=tie)


_W7 = dict(name="7A", t=(11.0, 13.8), bg=1, comps=[_cc("K7", 6.95, 7.5, 0.06, 1.2, 7.15, 0.3),
                                                     _cc("K7x", 6.95, 7.5, 0.45, 2.2, 7.25, 1.0)])
_W5 = dict(name="5A", t=(16.4, 19.6), bg=1, comps=[_cc("I5", 4.93, 5.06, 0.06, 1.2, 4.99, 0.3),
                                                     _cc("C47", 4.68, 4.80, 0.06, 1.0, 4.73, 0.3)])
_W35 = dict(name="35A", t=(24.45, 25.65), bg=1, comps=[_cc("K358", 3.565, 3.61, 0.05, 0.9, 3.58, 0.25),
                                                         _cc("C354", 3.52, 3.56, 0.05, 0.9, 3.54, 0.25, tie="K358")])
CLAY_WINDOWS = {
    "EG": [dict(name="low", t=(2.5, 10.7), bg=2, comps=[
        _cc("S17", 15.6, 18.2, 0.25, 2.6, 16.9, 0.8), _cc("ML", 10.6, 15.4, 0.6, 3.5, 12.5, 1.6),
        _cc("C14", 13.6, 14.8, 0.08, 1.0, 14.2, 0.3), _cc("I10", 9.7, 10.4, 0.06, 1.3, 10.0, 0.3)]), _W7, _W5, _W35],
    "AD": [dict(name="low", t=(2.5, 10.7), bg=2, comps=[
        _cc("E14", 11.5, 15.8, 0.3, 2.6, 14.5, 1.0), _cc("C14", 13.6, 14.8, 0.08, 1.0, 14.2, 0.3),
        _cc("I10", 9.7, 10.4, 0.06, 1.3, 10.0, 0.3)]), _W7, _W5, _W35],
    "H550": [dict(name="low", t=(2.5, 10.7), bg=2, comps=[
        _cc("C14", 13.4, 14.6, 0.08, 1.2, 14.0, 0.3), _cc("I10", 9.6, 10.6, 0.06, 2.2, 10.0, 0.4)]), _W7, _W5, _W35],
    "FA": [dict(name="low", t=(2.5, 10.7), bg=2, comps=[
        _cc("C14", 13.6, 14.8, 0.08, 1.0, 14.2, 0.3), _cc("H10", 10.15, 10.9, 0.1, 1.5, 10.4, 0.5),
        _cc("I10", 9.7, 10.15, 0.06, 1.0, 10.0, 0.3)]), _W7, _W5, _W35],
}
CLAY_WINDOWS["H350"] = CLAY_WINDOWS["H550"]

# 礦物縮寫依 IMA–CNMNC 建議（Warr, 2021）：Sme 膨潤石、Ilt 伊萊石、Kln 高嶺石、Chl 綠泥石、Vrm 蛭石、Hly 埃洛石、Qz 石英
CLAY_ABBR = {"S17": "Sme", "E14": "Sme/Vrm", "ML": "I/S", "I10": "Ilt", "I5": "Ilt", "C14": "Chl", "C47": "Chl", "C354": "Chl",
             "K7": "Kln+Chl", "K7x": "Kln+Chl", "K358": "Kln", "H10": "Hly"}
CLAY_COL = {"S17": "#E69F00", "E14": "#E69F00", "ML": "#D55E00", "I10": "#0072B2", "I5": "#0072B2", "C14": "#CC79A7",
            "C47": "#CC79A7", "C354": "#CC79A7", "K7": "#009E73", "K7x": "#009E73", "K358": "#009E73", "H10": "#8C564B"}
BISCAYE_COL = {"Smectite": "#E69F00", "Illite": "#0072B2", "Kaolinite": "#009E73", "Chlorite": "#CC79A7"}


def _pv_unit(x, c, w, eta):
    """面積 = 1 的 pseudo-Voigt（x、c、w 皆為 °2θ）"""
    z = (x - c) / w
    return (eta * (0.6366197723675814 / w) / (1.0 + 4.0 * z * z)
            + (1.0 - eta) * (0.9394372786996513 / w) * np.exp(-2.772588722239781 * z * z))


def _pv_ka(x, c, w, eta, ka2):
    """面積 = 1 的峰形；實驗室 Cu 光源（ka2=True）時為 Kα1 + ½ Kα2 雙峰，c 為 Kα1 的位置"""
    if not ka2:
        return _pv_unit(x, c, w, eta)
    c2 = 2.0 * math.degrees(math.asin(min(1.0, 1.544426 / TAG_WL * math.sin(math.radians(c / 2.0)))))
    return (_pv_unit(x, c, w, eta) + 0.5 * _pv_unit(x, c2, w, eta)) / 1.5


def _lsq_peaks_bg(D, y, k):
    """y ≈ D θ；前 k 個係數（峰面積）≥ 0，其餘（背景修正）不限"""
    cols = list(range(D.shape[1]))
    while True:
        th = np.linalg.lstsq(D[:, cols], y, rcond=None)[0]
        neg = [j for j, c in enumerate(cols) if c < k and th[j] < 0]
        if not neg:
            full = np.zeros(D.shape[1])
            full[cols] = th
            return full
        cols.pop(min(neg, key=lambda j: th[j]))


def _fit_window(x, yb, comps, bgdeg, ka2=False):
    """
    一個區間內的多峰擬合（Levenberg–Marquardt，峰面積與背景修正以線性最小平方直接求解）。
    回傳每個峰的中心、半高寬、面積與面積的標準差（由擬合的共變異矩陣估計）。
    """
    k = len(comps)
    xm, xs = float(x.mean()), max(float(x.max() - x.min()) / 2.0, 1e-6)
    Bg = np.stack([((x - xm) / xs) ** p for p in range(bgdeg + 1)], 1)
    free_w = [c["key"] for c in comps if not c.get("tie")]
    lo = [_t_of_d(c["dhi"]) for c in comps] + [next(c for c in comps if c["key"] == f)["wlo"] for f in free_w] + [0.0]
    hi = [_t_of_d(c["dlo"]) for c in comps] + [next(c for c in comps if c["key"] == f)["whi"] for f in free_w] + [1.0]
    lo, hi = np.array(lo), np.array(hi)

    def unpack(v):
        ws = dict(zip(free_w, v[k:k + len(free_w)]))
        return v[:k], np.array([ws[c["tie"] or c["key"]] for c in comps]), float(v[-1])

    def design(v):
        cen, wid, eta = unpack(v)
        return np.hstack([np.stack([_pv_ka(x, cen[i], wid[i], eta, ka2) for i in range(k)], 1), Bg])

    def tov(z):
        return lo + (hi - lo) * (1.0 + np.tanh(z)) / 2.0

    def toz(v):
        f = np.clip((np.asarray(v, float) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
        return np.arctanh(2.0 * f - 1.0)

    def resid(z):
        D = design(tov(z))
        th = _lsq_peaks_bg(D, yb, k)
        return yb - D @ th, th

    # 初始值：窄峰取區間內實際的最高點，寬峰用典型位置；另外試一組較寬的初始半高寬
    sm = np.convolve(yb, np.ones(5) / 5, mode="same") if len(yb) > 7 else yb
    cen0 = []
    for i, c in enumerate(comps):
        a, b = lo[i], hi[i]
        m = (x >= a) & (x <= b)
        if c["whi"] <= 1.3 and m.sum() >= 3:
            j = int(np.argmax(np.where(m, sm, -np.inf)))
            cen0.append(x[j] if 0 < j < len(x) - 1 and sm[j] > 0 else _t_of_d(c["d0"]))
        else:
            cen0.append(_t_of_d(c["d0"]))
    starts = []
    for f in (1.0, 1.7):
        w0 = [next(c for c in comps if c["key"] == fw)["w0"] * f for fw in free_w]
        starts.append(np.array(cen0 + w0 + [0.5]))
    best = None
    for v0 in starts:
        z = toz(np.clip(v0, lo + 1e-9, hi - 1e-9))
        r, th = resid(z)
        ssr, lam = float(r @ r), 1e-2
        for _it in range(80):
            J = np.empty((len(x), len(z)))
            for j in range(len(z)):
                dz = np.zeros(len(z))
                dz[j] = 1e-4
                J[:, j] = (resid(z + dz)[0] - r) / 1e-4
            g, H = J.T @ r, J.T @ J
            improved, rel = False, 0.0
            for _ in range(10):
                try:
                    stp = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
                except np.linalg.LinAlgError:
                    lam *= 4
                    continue
                r2, th2 = resid(z + stp)
                s2 = float(r2 @ r2)
                if s2 < ssr:
                    rel = (ssr - s2) / max(ssr, 1e-300)
                    z, r, th, ssr = z + stp, r2, th2, s2
                    lam = max(lam / 3.0, 1e-7)
                    improved = True
                    break
                lam *= 4
            if not improved or rel < 1e-8:
                break
        if best is None or ssr < best[0]:
            best = (ssr, z, th, r)
    ssr, z, th, r = best
    v = tov(z)
    cen, wid, eta = unpack(v)
    # 共變異矩陣：模型對所有參數（中心、半高寬、η、面積、背景）的 Jacobian
    D = design(v)
    act = [i for i in range(k) if th[i] > 0] + list(range(k, D.shape[1]))
    cols = [D[:, i] for i in act]
    for j in range(len(v)):
        h = (hi[j] - lo[j]) * 1e-5
        v2 = v.copy()
        v2[j] = v[j] + h
        cols.append(((design(v2) - D) @ th) / h)
    Jf = np.stack(cols, 1)
    n, p = len(x), Jf.shape[1]
    s2 = ssr / max(n - p, 1)
    try:
        cov = np.linalg.pinv(Jf.T @ Jf) * s2
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    sd_area = np.zeros(k)
    for pos, i in enumerate(act):
        if i < k:
            sd_area[i] = math.sqrt(max(cov[pos, pos], 0.0))
    out = []
    for i, c in enumerate(comps):
        out.append(dict(key=c["key"], t=float(cen[i]), d=float(_d_of_t(cen[i])), fwhm=float(wid[i]), eta=eta,
                        area=float(th[i]), sd=float(sd_area[i]),
                        height=float(th[i] * _pv_ka(np.array([cen[i]]), cen[i], wid[i], eta, ka2)[0])))
    return dict(comps=out, bgc=th[k:], xm=xm, xs=xs, deg=bgdeg, sigma=math.sqrt(s2), ssr=ssr, ka2=ka2)


def clay_fit_pattern(x, y, treat, ka2=False):
    """擬合一條定向片圖譜（x 為 Cu Kα1 的 2θ；ka2 = 實驗室 Cu 光源含 Kα2）。"""
    o = np.argsort(x)
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    m = (x >= 1.0) & (x <= 34.0) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 50 or x[-1] - x[0] < 5:
        raise UserError("數據點太少或 2θ 範圍太窄，無法分析黏土礦物（需要約 3–30° 2θ，Cu Kα）。")
    step = max(float(np.median(np.diff(x))), 1e-4)
    xg = np.arange(x[0], x[-1] + step / 2, step)
    yg = np.interp(xg, x, y)
    bg = snip_background(yg, max(3, int(round(2.5 / step))))
    yb = yg - bg
    noise = 1.4826 * float(np.median(np.abs(np.diff(yb)))) / math.sqrt(2) or 1e-12
    comps, wins, oor = {}, [], []
    for W in CLAY_WINDOWS[treat]:
        a, b = W["t"]
        a2, b2 = max(a, xg[0] + 0.15), min(b, xg[-1] - 0.1)
        # 峰的兩側要有足夠的數據（約 1 個半高寬，最多 1°）才能定出背景；不足的峰標為「超出範圍」
        cover = [c for c in W["comps"] if _t_of_d(c["d0"]) - min(1.0, c["w0"]) >= a2
                 and _t_of_d(c["d0"]) + min(1.0, c["w0"]) <= b2]
        cover = [c if not c.get("tie") or any(cc["key"] == c["tie"] for cc in cover) else dict(c, tie=None) for c in cover]
        oor += [c["key"] for c in W["comps"] if c not in cover and c["key"] not in [cc["key"] for cc in cover]]
        keys = [c["key"] for c in cover]
        if not cover or b2 - a2 < 0.4 * (b - a):
            oor += keys
            continue
        mk = (xg >= a2) & (xg <= b2)
        xw, yw = xg[mk], yb[mk]
        use = list(cover)

        def weak_of(res):
            w = [c["key"] for c in res["comps"]
                 if c["area"] <= 0 or c["area"] < 3.0 * c["sd"] or c["height"] < 3.0 * max(noise, res["sigma"] * 0.7)]
            # 共用半高寬的峰：被依附的峰若還在就保留
            return [kk for kk in w if not any(cc.get("tie") == kk and cc["key"] not in w for cc in use)]

        res = None
        for _round in range(3):
            res = _fit_window(xw, yw, use, W["bg"], ka2)
            weak = weak_of(res)
            if not weak:
                break
            use = [c if c.get("tie") not in weak else dict(c, tie=None) for c in use if c["key"] not in weak]
            if not use:
                res = None
                break
        if res is not None and weak_of(res):
            weak = weak_of(res)
            use = [c if c.get("tie") not in weak else dict(c, tie=None) for c in use if c["key"] not in weak]
            res = _fit_window(xw, yw, use, W["bg"], ka2) if use else None
        got = {c["key"]: c for c in res["comps"]} if res else {}
        for kk in keys:
            c = got.get(kk)
            if c is None:
                comps[kk] = dict(key=kk, present=False, area=0.0, sd=0.0, d=None, t=None, fwhm=None, height=0.0)
            else:
                c = dict(c)
                c["present"] = bool(c["area"] > 3.0 * c["sd"] and c["height"] > 3.0 * max(noise, res["sigma"] * 0.7))
                comps[kk] = c
        if res:
            wins.append(dict(name=W["name"], a=float(a2), b=float(b2), res=res, keys=[c["key"] for c in use]))
    for kk in oor:
        comps[kk] = dict(key=kk, present=False, area=0.0, sd=0.0, d=None, t=None, fwhm=None, height=0.0, oor=True)
    # 寬峰（膨潤石、混層）卡在參數邊界時，代表擬合的是背景彎曲而不是峰
    for key in ("S17", "E14", "ML", "I10"):
        c = comps.get(key)
        if c and c.get("present"):
            spec = next(cc for W in CLAY_WINDOWS[treat] for cc in W["comps"] if cc["key"] == key)
            if c["fwhm"] >= 0.97 * spec["whi"] or c["d"] <= spec["dlo"] * 1.005 or c["d"] >= spec["dhi"] * 0.995:
                c["present"] = False
    # 混層寬峰：容易和背景的彎曲混淆，必須至少是 17 Å 或 10 Å 峰的 1/4 才算
    ml = comps.get("ML")
    if ml and ml.get("present"):
        ref = max(comps.get("S17", {}).get("area", 0.0), comps.get("I10", {}).get("area", 0.0))
        ml["present"] = ml["area"] >= 0.25 * ref
    # 石英 3.34 Å 峰高 ÷ 最強的黏土底面峰高：遠大於 1 表示是全岩隨機粉末，不是黏土定向片
    tq = _t_of_d(3.343)
    mq = (xg > tq - 0.15) & (xg < tq + 0.15)
    hq = float(yb[mq].max()) if mq.any() else 0.0
    hc = max([c["height"] for k2, c in comps.items() if c.get("present") and k2 in ("S17", "E14", "ML", "C14", "I10", "K7")] or [0.0])
    qz_ratio = hq / hc if hc > 0 else (float("inf") if hq > 10 * noise else 0.0)
    return dict(treat=treat, x=xg, y=yg, bg=bg, noise=noise, comps=comps, wins=wins, ka2=ka2, qz_ratio=qz_ratio)


def _cp(fit, key):
    if not fit:
        return None
    c = fit["comps"].get(key)
    return c if c and c.get("present") else None


def _oor(fit, key):
    return bool(fit and fit["comps"].get(key, {}).get("oor"))


def clay_interpret(F):
    """依各處理的峰位變化判讀黏土礦物。F = {處理: 擬合結果}；回傳 [{zh, en, tier, why}]、注意事項"""
    EG, AD, H5, H3, FA = (F.get(k) for k in ("EG", "AD", "H550", "H350", "FA"))
    H = H5 or H3
    out, notes = [], []

    def add(key, zh, en, tier, why):
        out.append(dict(key=key, zh=zh, en=en, tier=tier, why=why))

    # 膨潤石（smectite）與混層
    s17, ml, e14 = _cp(EG, "S17"), _cp(EG, "ML"), _cp(AD, "E14")
    if s17:
        why = "EG %.1f Å" % s17["d"]
        if e14:
            why = "AD %.1f Å → %s（乙二醇處理後膨脹）" % (e14["d"], why)
        else:
            why += "（膨脹到約 17 Å 是膨潤石的特徵）"
        tier = "certain"
        if s17["d"] < 16.4:
            tier = "possible"
            why += "；膨脹不完全，可能是伊萊石/膨潤石混層或層間陽離子影響"
        hi10, ei10 = _cp(H, "I10"), _cp(EG, "I10")
        if H and hi10 and ei10 and hi10["area"] > 1.3 * ei10["area"]:
            why += "；加熱後塌陷到 10 Å"
        add("smectite", "膨潤石", "smectite", tier, why)
    elif e14 and not EG:
        add("smectite", "膨潤石或蛭石", "smectite or vermiculite", "check",
            "AD %.1f Å；需要 EG 圖譜才能判斷是否會膨脹" % e14["d"])
    elif EG and _oor(EG, "S17"):
        notes.append("EG 圖譜沒有涵蓋 17 Å（約 5° 2θ, Cu Kα），無法判斷膨潤石；10 Å 峰的指標也可能受膨潤石影響。"
                     "定向片建議從 2–3° 2θ 開始量測。")
    if ml:
        add("mixed", "混層黏土（如伊萊石/膨潤石 I/S）", "mixed-layer clay (e.g. I/S)", "possible",
            "EG 在 %.1f Å 有寬峰（介於 10 與 17 Å 之間）" % ml["d"])

    # 伊萊石／雲母
    eg10, ad10 = _cp(EG, "I10"), _cp(AD, "I10")
    if eg10 or ad10:
        parts = []
        if eg10 and ad10:
            parts.append("AD %.2f Å、EG %.2f Å（乙二醇處理不移動）" % (ad10["d"], eg10["d"]))
        else:
            parts.append("%.2f Å" % (eg10 or ad10)["d"])
        i5 = _cp(EG, "I5") or _cp(AD, "I5")
        if i5:
            parts.append("%.2f Å (002)" % i5["d"])
        shift_is = eg10 and ad10 and ad10["d"] - eg10["d"] >= 0.08
        if shift_is:
            parts[0] = "AD %.2f Å → EG %.2f Å" % (ad10["d"], eg10["d"])
        elif eg10 and ad10 and ad10["fwhm"] > 1.25 * eg10["fwhm"]:
            parts.append("AD 的 10 Å 峰比 EG 寬，可能夾有少量膨脹層")
        lost = ad10 and EG and not eg10 and not _oor(EG, "I10")
        if lost:
            parts.append("但 EG 沒有 10 Å 峰")
        add("illite", "伊萊石／雲母", "illite/mica",
            "possible" if lost else ("certain" if (i5 or (eg10 and ad10)) else "possible"), "；".join(parts))
        if lost:
            add("expandable", "膨脹性黏土（膨潤石層）", "expandable clay (smectite layers)", "check",
                "AD 的 %.2f Å 峰在乙二醇處理後消失，可能是含膨脹層的黏土移到更大的 d 值" % ad10["d"])
        if shift_is and not ml:
            add("mixed", "伊萊石/膨潤石混層（I/S，以伊萊石為主）", "illite-rich illite/smectite (I/S)", "possible",
                "10 Å 峰在 AD 偏向大 d 值且較寬（%.2f Å，半高寬 %.2f°），乙二醇處理後變回 %.2f Å（%.2f°）：含少量膨脹層"
                % (ad10["d"], ad10["fwhm"], eg10["d"], eg10["fwhm"]))

    # 綠泥石／蛭石：AD 的 14 Å 容易和膨潤石重疊，有 EG 時只看 EG
    if EG:
        c14, c354, c47 = _cp(EG, "C14"), _cp(EG, "C354"), _cp(EG, "C47")
    else:
        c14 = _cp(AD, "C14")
        c14 = c14 if c14 and c14["fwhm"] <= 0.8 else None
        c354, c47 = _cp(AD, "C354"), _cp(AD, "C47")
    if c14 or c354:
        why = []
        if c14:
            why.append("%.1f Å%s" % (c14["d"], "（EG 不膨脹）" if _cp(EG, "C14") else ""))
        if c47:
            why.append("%.2f Å" % c47["d"])
        if c354:
            why.append("%.2f Å" % c354["d"])
        if H:
            if _cp(H, "C14"):
                add("chlorite", "綠泥石", "chlorite", "certain", "、".join(why) + "；加熱後 14 Å 仍在")
            elif c14:
                add("vermiculite", "蛭石", "vermiculite", "possible", "、".join(why) + "；加熱後 14 Å 消失（塌陷）")
            else:
                add("chlorite", "綠泥石", "chlorite", "possible", "、".join(why))
        else:
            if c14 and c354:
                add("chlorite", "綠泥石", "chlorite", "certain" if c47 else "possible", "、".join(why))
            else:
                add("chlorite", "綠泥石或蛭石", "chlorite or vermiculite", "possible",
                    "、".join(why) + "；加熱到 550 °C 可以區分")

    # 高嶺石／埃洛石
    k7 = _cp(EG, "K7") or _cp(AD, "K7")
    k7x = _cp(EG, "K7x") or _cp(AD, "K7x")
    k358 = _cp(EG, "K358") or _cp(AD, "K358")
    if k7 or k358 or k7x:
        why = []
        if k7 or k7x:
            why.append("%.2f Å" % (k7 or k7x)["d"])
        if k358:
            why.append("%.2f Å (002)" % k358["d"])
        if H5 and not (_cp(H5, "K7") or _cp(H5, "K7x")):
            add("kaolinite", "高嶺石", "kaolinite", "certain", "、".join(why) + "；550 °C 加熱後 7 Å 峰消失")
        elif H5 and not k358:
            add("kaolinite", "高嶺石（可能）", "kaolinite (possible)", "check",
                "、".join(why) + "；550 °C 加熱後 7 Å 仍有峰，可能多為綠泥石")
        elif k358:
            add("kaolinite", "高嶺石", "kaolinite", "certain", "、".join(why))
        elif not (c14 or c354):
            add("kaolinite", "高嶺石", "kaolinite", "possible", "、".join(why) + "（沒有綠泥石的峰）")
        else:
            add("kaolinite", "高嶺石（可能）", "kaolinite (possible)", "check",
                "、".join(why) + "；7 Å 也可能是綠泥石的 002，3.58 Å 峰不明顯")
        h10 = _cp(FA, "H10")
        kk = k7 or k7x
        if FA and h10:
            add("halloysite", "埃洛石", "halloysite", "certain", "甲醯胺處理後出現 %.1f Å 峰（7 Å 層膨脹）" % h10["d"])
        elif FA:
            notes.append("甲醯胺處理後沒有 10.4 Å 峰：7 Å 礦物以高嶺石為主。")
        else:
            broad = k7x if (k7 and k7x and k7x["fwhm"] > 0.6 and k7x["d"] >= 7.15 and k7x["area"] >= 0.5 * k7["area"]) else None
            why_h = []
            if kk and kk["fwhm"] > 0.55:
                why_h.append("7 Å 峰偏寬（半高寬 %.2f°）" % kk["fwhm"])
            if kk and kk["d"] >= 7.25:
                why_h.append("7 Å 峰的 d 值偏大（%.2f Å）" % kk["d"])
            if broad:
                why_h.append("7 Å 峰帶有寬的成分（%.2f Å，半高寬 %.2f°）" % (broad["d"], broad["fwhm"]))
            if why_h:
                add("halloysite", "埃洛石（可能）", "halloysite (possible)", "check",
                    "、".join(why_h) + "；甲醯胺測試可以確認")
    if not out:
        notes.append("沒有偵測到明顯的黏土礦物底面峰。")
    return out, notes


def clay_biscaye(F, interp, seed=0):
    """Biscaye (1965)：EG 圖譜 17 Å ×1、10 Å ×4、7 Å ×2；高嶺石／綠泥石依 3.58／3.54 Å 面積比分配。"""
    EG = F.get("EG")
    if not EG:
        return dict(ok=False, why="需要 EG（乙二醇）圖譜。")
    if _oor(EG, "S17") or _oor(EG, "I10") or _oor(EG, "K7"):
        return dict(ok=False, why="EG 圖譜沒有涵蓋 17、10、7 Å 三個峰的範圍（約 4–14° 2θ, Cu Kα）。")

    def A(*keys):
        a = s = 0.0
        for kk in keys:
            c = _cp(EG, kk)
            if c:
                a += c["area"]
                s = math.hypot(s, c["sd"])
        return a, s

    aS, aI, a7 = A("S17"), A("I10"), A("K7", "K7x")
    aK, aC = A("K358"), A("C354")
    warn = []
    has_chl = any(m["key"] in ("chlorite", "vermiculite") for m in interp)
    has_kao = any(m["key"] in ("kaolinite", "halloysite") for m in interp)
    if aK[0] + aC[0] > 0:
        split = None
    elif _oor(EG, "K358"):
        split = 0.5 if (has_chl and has_kao) else (0.0 if has_chl else 1.0)
        warn.append("EG 圖譜沒有涵蓋 3.58／3.54 Å（約 25° 2θ），高嶺石與綠泥石無法分開。")
    else:
        split = 0.5 if (has_chl and has_kao) else (0.0 if has_chl else 1.0)
        if has_chl and has_kao:
            warn.append("3.58／3.54 Å 峰太弱，高嶺石與綠泥石暫以各半分配。")
    rng = np.random.default_rng(seed)
    N = 4000

    def draw(v):
        return np.clip(v[0] + v[1] * rng.standard_normal(N), 0.0, None)

    def comp(S_, I_, K7_, k_, c_):
        if split is None:
            fr = np.where(k_ + c_ > 0, k_ / np.maximum(k_ + c_, 1e-300), 0.5)
        else:
            fr = split
        W = np.array([S_, 4.0 * I_, 2.0 * K7_ * fr, 2.0 * K7_ * (1 - fr)], dtype=float)
        tot = W.sum(0)
        return 100.0 * W / np.where(tot > 0, tot, np.nan)

    nom = comp(aS[0], aI[0], a7[0], aK[0], aC[0])
    if not np.all(np.isfinite(nom)):
        return dict(ok=False, why="EG 圖譜上沒有偵測到 17、10、7 Å 的峰。")
    mc = comp(draw(aS), draw(aI), draw(a7), draw(aK), draw(aC))
    sd = np.nanstd(mc, axis=1)
    if _cp(EG, "ML") and _cp(EG, "ML")["area"] > 0.2 * max(aS[0], 1e-30):
        warn.append("有混層黏土的寬峰（未計入四種礦物的比例）。")
    bulk = EG.get("qz_ratio", 0) > 3
    if bulk:
        warn.insert(0, "石英的 3.34 Å 峰遠強於黏土的底面峰（約 %.0f 倍），這看起來是全岩隨機粉末而不是 <2 µm 黏土定向片。"
                       "Biscaye 半定量是為定向片設計的，這裡的比例只能當參考；全岩的含量請用「全岩粉末」的 RIR 半定量。"
                    % min(EG["qz_ratio"], 999))
    names = ["Smectite", "Illite", "Kaolinite", "Chlorite"]
    return dict(ok=True, pct={n: float(v) for n, v in zip(names, nom)}, sd={n: float(v) for n, v in zip(names, sd)},
                areas=dict(S17=aS[0], I10=aI[0], K7=a7[0], K358=aK[0], C354=aC[0]), warn=warn, bulk=bulk)


def clay_indices(F):
    EG, AD = F.get("EG"), F.get("AD")
    out = {}
    if _cp(AD, "I10"):
        out["KI_AD"] = _cp(AD, "I10")["fwhm"]
    if _cp(EG, "I10"):
        out["KI_EG"] = _cp(EG, "I10")["fwhm"]
    i5, i10 = _cp(EG, "I5"), _cp(EG, "I10")
    if i5 and i10:
        r = i5["area"] / i10["area"]
        out["Esq"] = r
        out["Esq_sd"] = r * math.hypot(i5["sd"] / i5["area"], i10["sd"] / i10["area"])
    if _cp(EG, "S17"):
        out["Sm_FWHM"] = _cp(EG, "S17")["fwhm"]
    return out


def _clay_model(fit, w):
    """某一區間的擬合曲線（含背景）與各峰的曲線"""
    x = fit["x"]
    m = (x >= w["a"]) & (x <= w["b"])
    xw = x[m]
    res = w["res"]
    u = (xw - res["xm"]) / res["xs"]
    bgc = sum(res["bgc"][p] * u ** p for p in range(res["deg"] + 1)) + fit["bg"][m]
    parts = []
    for c in res["comps"]:
        if c["area"] > 0:
            parts.append((c["key"], c["area"] * _pv_ka(xw, c["t"], c["fwhm"], c["eta"], res.get("ka2", False))))
    total = bgc + (np.sum([p for _, p in parts], 0) if parts else 0)
    return xw, bgc, parts, total


def _quartz_peaks(fit):
    """石英 4.26 Å 與 3.34 Å 兩根峰都明顯存在時，回傳它們的 2θ（只用來在圖上標示 Q）"""
    x, yb, nz = fit["x"], fit["y"] - fit["bg"], fit["noise"]
    pos = []
    for d in (4.257, 3.343):
        t = _t_of_d(d)
        m = (x > t - 0.12) & (x < t + 0.12)
        side = ((x > t - 0.6) & (x < t - 0.3)) | ((x > t + 0.3) & (x < t + 0.6))
        if m.sum() < 3 or side.sum() < 3:
            return []
        j = int(np.argmax(np.where(m, yb, -np.inf)))
        if yb[j] < 8 * nz + np.median(yb[side]) or not (yb[j] >= yb[max(j - 1, 0)] and yb[j] >= yb[min(j + 1, len(x) - 1)]):
            return []
        pos.append(float(x[j]))
    return pos


def clay_figure(sample, S=None, fmt="png", dpi=300, width_mm=180):
    """單一樣品各處理的圖譜（上到下 AD、EG、加熱、FA），標出擬合的底面峰（期刊風格，英文標示）。"""
    S = merge_settings(S or {})
    fits = [sample["fits"][t] for t in CLAY_ORDER if t in sample["fits"]]
    n = len(fits)
    W = width_mm / 25.4
    H = W * (0.30 + 0.17 * n)
    with matplotlib.rc_context(make_rc(S)):
        fig = Figure(figsize=(W, H), dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0.07, 0.11, 0.86, 0.80])
        lo = min(max(f["x"][0], 2.5) for f in fits)
        hi = max(min(f["x"][-1], 30.0) for f in fits)
        lw = S["lw"]
        GAP = 1.38
        for i, f in enumerate(fits):
            off = (n - 1 - i) * GAP
            x, y = f["x"], f["y"]
            m = (x >= lo) & (x <= hi)
            ref = (x >= max(lo, 3.5)) & (x <= hi)
            clay = (x >= max(lo, 3.5)) & (x <= 14.0)
            sc_all = float(np.max(y[ref])) if ref.sum() > 5 else float(y[m].max())
            sc = float(np.max(y[clay])) if clay.sum() > 5 else sc_all
            sc = sc if sc > 0 else (sc_all if sc_all > 0 else 1.0)
            # 全岩粉末的石英峰常遠強於黏土峰：以 4–14° 的黏土區間定比例，超出的峰截斷（圖中以 // 標示）
            yy_n = y[m] / sc
            clip = 1.12
            ax.plot(x[m], np.minimum(yy_n, clip) + off, color="black", lw=lw * 0.8, zorder=3)
            over = yy_n > clip
            if over.any():
                idx = np.where(over)[0]
                runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                for r_ in runs:
                    xm_ = float(x[m][r_].mean())
                    ax.text(xm_, off + clip, "//", ha="center", va="center", fontsize=S["fs_tick"] * 0.8, rotation=0,
                            zorder=5, bbox=dict(boxstyle="square,pad=0.02", fc="white", ec="none"))
            for w in f["wins"]:
                xw, bgc, parts, total = _clay_model(f, w)
                for key, p in parts:
                    if not f["comps"].get(key, {}).get("present"):
                        continue
                    ax.fill_between(xw, bgc / sc + off, (bgc + p) / sc + off, color=CLAY_COL.get(key, "#999999"),
                                    alpha=0.45, lw=0, zorder=1)
            ax.text(0.003, off + 0.55, CLAY_TREAT_FIG[f["treat"]], transform=blended_transform_factory(ax.transAxes, ax.transData),
                    ha="left", va="bottom", fontsize=S["fs_legend"], fontweight="bold")
            # 峰位標示：d 值（Å）與礦物縮寫
            lab = []
            for key in ("S17", "E14", "ML", "C14", "H10", "I10", "K7", "I5"):
                c = _cp(f, key)
                if c and lo < c["t"] < hi:
                    lab.append((c["t"], "%.1f" % c["d"] if c["d"] >= 9.5 else "%.2f" % c["d"], CLAY_ABBR[key]))
            k3, c3 = _cp(f, "K358"), _cp(f, "C354")
            if k3 and c3:
                lab.append(((k3["t"] + c3["t"]) / 2, "3.58/3.54", "Kln/Chl"))
            elif k3 or c3:
                cc = k3 or c3
                lab.append((cc["t"], "%.2f" % cc["d"], "Kln" if k3 else "Chl"))
            lab += [(t, "%.2f" % _d_of_t(t), "Qz") for t in _quartz_peaks(f) if lo < t < hi]
            # 太靠近的標示合併成一個（例如 AD 的 14.6 Å 膨潤石與 14.2 Å 綠泥石）
            lab.sort()
            merged = []
            for t, dtxt, ab in lab:
                if merged and t - merged[-1][3] < 0.8:
                    m0 = merged[-1]
                    merged[-1] = ((m0[0] + t) / 2, m0[1] + "/" + dtxt,
                                  m0[2] if ab in m0[2].split("/") else m0[2] + "/" + ab, t)
                else:
                    merged.append((t, dtxt, ab, t))
            for t, dtxt, ab, _t2 in merged:
                j = int(np.argmin(np.abs(x - t)))
                yy = float(np.max(y[max(0, j - 12):j + 13])) / sc + off
                ax.annotate(dtxt + "\n" + ab, (t, min(yy, off + 1.12)), xytext=(0, 1.5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=S["fs_tick"] * 0.85, linespacing=0.95, zorder=4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.06, (n - 1) * GAP + 1.42)
        ax.set_yticks([])
        ax.set_xlabel("2θ (°, Cu Kα)", fontsize=S["fs_label"])
        ax.set_ylabel("Intensity (a.u.)", fontsize=S["fs_label"])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.tick_params(axis="x", which="both", top=False)
        sec = ax.secondary_xaxis("top", functions=(lambda t: TAG_WL / (2 * np.sin(np.radians(np.clip(t, 0.5, 170) / 2))),
                                                    lambda d: 2 * np.degrees(np.arcsin(np.clip(TAG_WL / (2 * np.clip(d, 0.5, None)), 0, 1)))))
        sec.set_xticks([v for v in (17, 14, 10, 7, 5, 4, 3.5) if lo < _t_of_d(v) < hi])
        sec.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: ("%.1f" % v).rstrip("0").rstrip(".")))
        sec.set_xlabel("d (Å)", fontsize=S["fs_label"])
        sec.tick_params(labelsize=S["fs_tick"])
        handles = [Line2D([0], [0], color=BISCAYE_COL[k], lw=5, alpha=0.6) for k in ("Smectite", "Illite", "Kaolinite", "Chlorite")]
        ax.legend(handles, ["Sme smectite", "Ilt illite", "Kln kaolinite", "Chl chlorite"], loc="upper right", frameon=False,
                  fontsize=S["fs_legend"] * 0.9, ncol=4, handlelength=1.0, columnspacing=1.0, borderaxespad=0.2)
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    return buf.getvalue()


def clay_bar_figure(samples, S=None, fmt="png", dpi=300, width_mm=89):
    """Biscaye 半定量結果的橫向堆疊長條圖（每個樣品一列）"""
    S = merge_settings(S or {})
    rows = [s for s in samples if s.get("bisc", {}).get("ok")]
    if not rows:
        raise UserError("沒有可畫的半定量結果（需要 EG 圖譜）。")
    names = ["Smectite", "Illite", "Kaolinite", "Chlorite"]
    W = width_mm / 25.4
    H = 0.55 + 0.17 * len(rows) + 0.35
    with matplotlib.rc_context(make_rc(S)):
        fig = Figure(figsize=(W, H), dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        for i, s in enumerate(rows):
            left = 0.0
            for nm in names:
                v = s["bisc"]["pct"][nm]
                ax.barh(i, v, left=left, height=0.7, color=BISCAYE_COL[nm], edgecolor="white", lw=0.3)
                if v >= 8:
                    ax.text(left + v / 2, i, "%.0f" % v, ha="center", va="center", fontsize=S["fs_tick"] * 0.85,
                            color="white" if nm in ("Illite", "Kaolinite") else "black")
                left += v
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([s["sample"] for s in rows], fontsize=S["fs_tick"])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Relative abundance (%)", fontsize=S["fs_label"])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend([Line2D([0], [0], color=BISCAYE_COL[n], lw=6) for n in names], names, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=S["fs_legend"], handlelength=1.0,
                  columnspacing=0.9)
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    return buf.getvalue()


def _f(v, nd=2):
    return "" if v is None else ("%.*f" % (nd, v))


def clay_csv(samples):
    """結果表（每個樣品一列）與峰擬合明細"""
    o = io.StringIO()
    w = csv.writer(o)
    w.writerow(["sample", "treatments", "smectite_%", "smectite_sd", "illite_%", "illite_sd", "kaolinite_%", "kaolinite_sd",
                "chlorite_%", "chlorite_sd", "Kubler_index_AD_deg2theta", "Kubler_index_EG_deg2theta",
                "Esquevin_index_5A_10A", "Esquevin_sd", "smectite_FWHM_EG_deg2theta", "minerals", "notes"])
    for s in samples:
        b, ix = s["bisc"], s["indices"]
        p, sd = (b.get("pct", {}), b.get("sd", {})) if b.get("ok") else ({}, {})
        w.writerow([s["sample"], " ".join(t for t in CLAY_ORDER if t in s["fits"])]
                   + sum([[_f(p.get(n), 1), _f(sd.get(n), 1)] for n in ("Smectite", "Illite", "Kaolinite", "Chlorite")], [])
                   + [_f(ix.get("KI_AD"), 3), _f(ix.get("KI_EG"), 3), _f(ix.get("Esq"), 3), _f(ix.get("Esq_sd"), 3),
                      _f(ix.get("Sm_FWHM"), 3), "; ".join(m["en"] for m in s["minerals"]),
                      " | ".join(b.get("warn", []) + ([b["why"]] if not b.get("ok") else []))])
    p = io.StringIO()
    w2 = csv.writer(p)
    w2.writerow(["sample", "treatment", "peak", "mineral", "d_A", "two_theta_CuKa1", "FWHM_deg2theta", "area", "area_sd", "detected"])
    for s in samples:
        for t in CLAY_ORDER:
            f = s["fits"].get(t)
            if not f:
                continue
            for kk, c in f["comps"].items():
                w2.writerow([s["sample"], t, kk, CLAY_ABBR.get(kk, ""), _f(c.get("d"), 3), _f(c.get("t"), 3),
                             _f(c.get("fwhm"), 3), _f(c.get("area"), 4), _f(c.get("sd"), 4),
                             "out of range" if c.get("oor") else ("yes" if c.get("present") else "no")])
    return o.getvalue(), p.getvalue()


CLAY_REFS = {
    "biscaye": "Biscaye, P.E., 1965. Mineralogy and sedimentation of recent deep-sea clay in the Atlantic Ocean and adjacent seas and oceans. Geological Society of America Bulletin 76, 803–832.",
    "mr": "Moore, D.M., Reynolds, R.C., Jr., 1997. X-Ray Diffraction and the Identification and Analysis of Clay Minerals, 2nd ed. Oxford University Press, New York.",
    "kubler": "Kübler, B., 1964. Les argiles, indicateurs de métamorphisme. Revue de l'Institut Français du Pétrole 19, 1093–1112.",
    "warr": "Warr, L.N., Rice, A.H.N., 1994. Interlaboratory standardization and calibration of clay mineral crystallinity and crystallite size data. Journal of Metamorphic Geology 12, 141–152.",
    "esq": "Esquevin, J., 1969. Influence de la composition chimique des illites sur leur cristallinité. Bulletin du Centre de Recherches de Pau-SNPA 3, 147–153.",
    "snip": "Ryan, C.G., Clayton, E., Griffin, W.L., Sie, S.H., Cousens, D.R., 1988. SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications. Nuclear Instruments and Methods in Physics Research B 34, 396–402.",
    "church": "Churchman, G.J., Whitton, J.S., Claridge, G.G.C., Theng, B.K.G., 1984. Intercalation method using formamide for differentiating halloysite from kaolinite. Clays and Clay Minerals 32, 241–248.",
}


def clay_methods(samples, sync_wl=None):
    treats = [t for t in CLAY_ORDER if any(t in s["fits"] for s in samples)]
    tr = [CLAY_TREAT_EN[t] for t in treats]
    tr_txt = ", ".join(tr[:-1]) + (" and " if len(tr) > 1 else "") + tr[-1] if tr else "untreated"
    has_bisc = any(s["bisc"].get("ok") for s in samples)
    has_ki = any("KI_AD" in s["indices"] or "KI_EG" in s["indices"] for s in samples)
    has_esq = any("Esq" in s["indices"] for s in samples)
    refs = ["mr", "snip"]
    p = ["Clay minerals were identified from the basal reflections of oriented mounts measured %s "
         "(Moore and Reynolds, 1997)." % tr_txt]
    if sync_wl:
        p.append("Synchrotron patterns (λ = %s Å) were converted to the 2θ scale of Cu Kα1 radiation (λ = 1.540598 Å) "
                 "with Bragg's law before analysis." % wl(sync_wl))
    p.append("After subtraction of a SNIP background (Ryan et al., 1988), the basal reflections were decomposed into "
             "pseudo-Voigt profiles by non-linear least-squares fitting, and peak positions were converted to d-spacings.")
    if "FA" in treats:
        refs.append("church")
        p.append("Halloysite was distinguished from kaolinite by formamide intercalation (Churchman et al., 1984).")
    if has_bisc:
        refs.append("biscaye")
        p.append("Semi-quantitative proportions of smectite, illite, kaolinite and chlorite were estimated from the EG "
                 "patterns using the integrated areas of the 17 Å (smectite), 10 Å (illite) and 7 Å (kaolinite + chlorite) "
                 "reflections weighted by factors of 1, 4 and 2, respectively (Biscaye, 1965); kaolinite and chlorite "
                 "were apportioned according to the area ratio of their 3.58 Å and 3.54 Å reflections. The four groups "
                 "were assumed to sum to 100%. Reported uncertainties (1σ) reflect peak-fitting errors only and do not "
                 "include sample-preparation effects.")
    if has_ki:
        refs += ["kubler", "warr"]
        p.append("Illite crystallinity (Kübler index) is given as the full width at half maximum of the 10 Å reflection "
                 "in Δ°2θ (Cu Kα) (Kübler, 1964), without calibration to the Crystallinity Index Standard scale "
                 "(Warr and Rice, 1994).")
    if has_esq:
        refs.append("esq")
        p.append("The illite chemistry index was calculated as the ratio of the 5 Å to 10 Å peak areas on the EG "
                 "patterns (Esquevin, 1969).")
    p.append("Data processing was carried out with XRD Studio v%s." % APP_VERSION)
    order = ["biscaye", "church", "esq", "kubler", "mr", "snip", "warr"]
    return " ".join(p) + "\n\nReferences\n" + "\n".join(CLAY_REFS[r] for r in order if r in refs)


def _png_data_url(b):
    import base64
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def clay_run(S, items, a):
    """分析多個樣品；a["groups"] = [{sample, files: [{id, treat}]}]"""
    groups = a.get("groups") or []
    if not groups:
        raise UserError("請先指定每個檔案的樣品與處理方式。")
    samples, sync_wl = [], None
    for g in groups:
        fits, files = {}, {}
        for f in g.get("files") or []:
            iid, treat = f.get("id"), f.get("treat")
            if iid not in DATA or treat not in CLAY_WINDOWS:
                continue
            if treat in fits:
                raise UserError("樣品「%s」有兩個「%s」檔案，請只保留一個。" % (g.get("sample"), CLAY_TREAT_ZH[treat]))
            d0 = DATA[iid]
            eff = effective(d0, S)
            if eff["type"] != "inhouse" and eff["convert"]:
                sync_wl = eff["src"]
            it = next((i for i in items if i.get("id") == iid), {})
            x, y, _ = converted_xy(d0, eff)
            x = x + (fnum(it.get("shift"), 0.0) or 0.0)
            fits[treat] = clay_fit_pattern(x, y, treat, ka2=eff["type"] == "inhouse")
            files[treat] = dict(id=iid, name=d0["name"], sync=eff["type"] != "inhouse")
        if not fits:
            continue
        minerals, notes = clay_interpret(fits)
        bisc = clay_biscaye(fits, minerals)
        ix = clay_indices(fits)
        if any(v["sync"] for v in files.values()):
            notes.append("同步輻射數據：Biscaye 權重與 Kübler 指數是以實驗室 Cu 繞射儀建立的，數值不宜直接和文獻比較。")
        samples.append(dict(sample=str(g.get("sample") or "sample"), fits=fits, files=files, minerals=minerals,
                            notes=notes, bisc=bisc, indices=ix))
    if not samples:
        raise UserError("沒有可分析的檔案。")
    CLAY_LAST.clear()
    CLAY_LAST.update(samples=samples, sync_wl=sync_wl)
    out = []
    for s in samples:
        comps = {t: [dict(key=kk, abbr=CLAY_ABBR.get(kk, ""), d=c.get("d"), t=c.get("t"), fwhm=c.get("fwhm"),
                          area=c.get("area"), sd=c.get("sd"), present=bool(c.get("present")), oor=bool(c.get("oor")))
                     for kk, c in f["comps"].items()] for t, f in s["fits"].items()}
        out.append(dict(sample=s["sample"], treats=[t for t in CLAY_ORDER if t in s["fits"]], files=s["files"],
                        minerals=s["minerals"], notes=s["notes"], bisc=s["bisc"], indices=s["indices"], comps=comps,
                        fig=_png_data_url(clay_figure(s, S, "png", dpi=110, width_mm=150))))
    return dict(samples=out, methods=clay_methods(samples, sync_wl))


def clay_export(S, a):
    """下載：結果表、峰擬合明細、方法說明、每個樣品的圖（PNG + PDF）、組成長條圖"""
    samples = CLAY_LAST.get("samples")
    if not samples:
        raise UserError("請先執行黏土分析。")
    res, peaks = clay_csv(samples)
    files = [("clay_results.csv", "﻿" + res), ("clay_peak_fits.csv", "﻿" + peaks),
             ("methods.txt", clay_methods(samples, CLAY_LAST.get("sync_wl")))]
    for s in samples:
        nm = safe_name(s["sample"], "sample")
        files.append(("figures/%s.png" % nm, clay_figure(s, S, "png", dpi=600)))
        files.append(("figures/%s.pdf" % nm, clay_figure(s, S, "pdf")))
    if any(s["bisc"].get("ok") for s in samples):
        files.append(("figures/composition.png", clay_bar_figure(samples, S, "png", dpi=600)))
        files.append(("figures/composition.pdf", clay_bar_figure(samples, S, "pdf")))
    return {"name": "clay_minerals.zip", "mime": "application/zip"}, make_zip(files)


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
    if cmd == "rv_package":
        return rietveld_package(merge_settings(S_in), items, a)
    if cmd == "rv_load":
        return rv_load(a), None
    if cmd == "rv_render":
        S = merge_settings(S_in)
        if a.get("export"):
            fmt = S["fmt"]
            dpi = float(S["dpi"]) if fmt in ("png", "tiff") else 300
            data, info = rietveld_figure(S_in, a.get("opts") or {}, fmt, dpi)
            return dict(info, name=safe_name(S["fig_name"]) + "_Rietveld." + FMT_EXT[fmt], mime=FMT_MIME[fmt]), data
        data, info = rietveld_figure(S_in, a.get("opts") or {}, "png", 160)
        return dict(info, fig=_png_data_url(data)), None
    if cmd == "quant_export":
        return quant_export(merge_settings(S_in), a)
    if cmd == "clay_run":
        return clay_run(merge_settings(S_in), items, a), None
    if cmd == "clay_export":
        return clay_export(merge_settings(S_in), a)
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
