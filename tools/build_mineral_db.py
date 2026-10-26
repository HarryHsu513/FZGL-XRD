# -*- coding: utf-8 -*-
"""
建立 XRD Studio「鑑定」功能用的礦物參考資料庫  →  minerals.json.gz（和 index.html 同一層）

資料來源（RRUFF 網站提供的整包下載）：
  * RRUFF 粉末 XRD：DIF 檔（以實測樣品精修的晶胞計算的繞射峰表）
      https://www.rruff.net/zipped_data_files/powder/DIF.zip
  * AMCSD（American Mineralogist Crystal Structure Database）：DIF 峰表 + AMC 結構（取原子種類）
      https://www.rruff.net/AMS/zipped_files/dif.zip
      https://www.rruff.net/AMS/zipped_files/amc.zip
引用：
  Lafuente B, Downs R T, Yang H, Stone N (2015) The power of databases: the RRUFF project.
  Downs R T, Hall-Wallace M (2003) The American Mineralogist Crystal Structure Database. Am. Mineral. 88, 247-250.

RRUFF 更新資料時重新執行即可：
    python tools/build_mineral_db.py              # 自動下載到 tools/_cache 後建立
    python tools/build_mineral_db.py 資料夾        # 使用已下載的 DIF.zip / dif.zip / amc.zip

篩選規則：
  * RRUFF：每種礦物最多保留 3 個樣品。
  * AMCSD：只保留常溫常壓的結構（排除標示高壓、高溫或低溫者），同條件時優先較新的結構；
    RRUFF 已有的礦物補 1 筆，RRUFF 沒有的礦物保留 2 筆；全部都不是常溫常壓時保留 1 筆並註明條件。
  * 每筆保留最強的 40 根峰（相對強度 ≥ 1），以 d 值儲存（與波長無關）。
"""
import gzip
import json
import math
import os
import re
import sys
import urllib.request
import zipfile
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "minerals.json.gz")
URLS = {
    "DIF.zip": "https://www.rruff.net/zipped_data_files/powder/DIF.zip",
    "dif.zip": "https://www.rruff.net/AMS/zipped_files/dif.zip",
    "amc.zip": "https://www.rruff.net/AMS/zipped_files/amc.zip",
}
ELEM = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os
Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr""".split())
PEAK = re.compile(r"^\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")
LAM = 1.541838  # DIF 檔計算用的波長；只用來合併極近的峰，最後以 d 值輸出


def get_zips(folder):
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for name, url in URLS.items():
        p = os.path.join(folder, name)
        alt = os.path.join(folder, {"DIF.zip": "rruff_DIF.zip", "dif.zip": "amcsd_dif.zip",
                                    "amc.zip": "amcsd_amc.zip"}[name])
        if os.path.exists(alt):
            p = alt
        if not os.path.exists(p):
            print("下載", url)
            urllib.request.urlretrieve(url, p)
        paths[name] = p
    return paths


def elem_of(label):
    m = re.match(r"([A-Z][a-z]?)", label)
    if not m:
        return None
    s = m.group(1)
    if s == "D":
        return "H"
    if s in ELEM:
        return s
    return s[0] if s[0] in ELEM else None


def peaks_of(text):
    out = []
    for ln in text.splitlines():
        m = PEAK.match(ln)
        if m:
            h = [int(m.group(k)) for k in (4, 5, 6)]
            hkl = "".join(str(v) for v in h) if all(abs(v) < 10 for v in h) else " ".join(str(v) for v in h)
            out.append((float(m.group(3)), float(m.group(2)), hkl))
    return out


def compact(peaks, keep=40, imin=1.0):
    """合併 2θ 相差 < 0.03° 的反射（加總強度），重新正規化，保留最強的 keep 根。"""
    rows = []
    for d, I, h in peaks:
        s = LAM / (2 * d)
        if s < 1:
            rows.append((2 * math.degrees(math.asin(s)), d, I, h))
    rows.sort()
    merged = []
    for t, d, I, h in rows:
        if merged and t - merged[-1][0] < 0.03:
            a = merged[-1]
            tot = a[2] + I
            merged[-1] = ((a[0] * a[2] + t * I) / tot, (a[1] * a[2] + d * I) / tot, tot, a[3] if a[2] >= I else h)
        else:
            merged.append((t, d, I, h))
    if not merged:
        return ""
    mx = max(m[2] for m in merged)
    keep_rows = sorted([m for m in merged if 100 * m[2] / mx >= imin], key=lambda m: -m[2])[:keep]
    keep_rows.sort(key=lambda m: -m[1])
    return ";".join("%.4f:%d:%s" % (m[1], max(1, round(100 * m[2] / mx)), m[3]) for m in keep_rows)


NONAMB = re.compile(r"(\d+(\.\d+)?)\s*(GPa|kbar|MPa)|T\s*=\s*(\d+(\.\d+)?)\s*(K|deg|°|C\b)", re.I)


def condition(lines):
    """回傳 (是否常溫常壓, 條件說明)。"""
    txt = " ".join(lines)
    note = []
    amb = True
    for m in NONAMB.finditer(txt):
        if m.group(3):
            v = float(m.group(1))
            unit = m.group(3).lower()
            gpa = v if unit == "gpa" else v / 10 if unit == "kbar" else v / 1000
            if gpa > 0.01:
                amb = False
                note.append("P = %s %s" % (m.group(1), m.group(3)))
        else:
            v = float(m.group(4))
            k = v if m.group(6).upper().startswith("K") else v + 273
            if not 263 <= k <= 313:
                amb = False
                note.append("T = %s %s" % (m.group(4), "K" if m.group(6).upper().startswith("K") else "°C"))
    if re.search(r"high[- ]pressure|high[- ]temperature", txt, re.I) and not note:
        amb = False
        note.append("高溫或高壓")
    return amb, ", ".join(dict.fromkeys(note))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_cache")
    z = get_zips(folder)
    entries = []

    # —— RRUFF 樣品
    zr = zipfile.ZipFile(z["DIF.zip"])
    per_name = {}
    n_rruff = {}
    for n in zr.namelist():
        k = n.split("__")[0].lower()
        n_rruff[k] = n_rruff.get(k, 0) + 1
    for n in sorted(zr.namelist()):
        t = zr.read(n).decode("latin-1")
        lines = t.splitlines()
        name = lines[0].strip()
        rid = n.split("__")[1] if "__" in n else ""
        if per_name.get(name.lower(), 0) >= 3:
            continue
        els, on = set(), False
        for ln in lines:
            if "ATOM" in ln and "OCCUPANCY" in ln:
                on = True
                continue
            if on:
                if not ln.strip():
                    if els:
                        on = False
                    continue
                if "WAVELENGTH" in ln:
                    on = False
                    continue
                e = elem_of(ln.split()[0])
                if e:
                    els.add(e)
        loc = next((ln.split(":", 1)[1].strip() for ln in lines[:12] if "Locality:" in ln), "")
        p = compact(peaks_of(t))
        if p:
            per_name[name.lower()] = per_name.get(name.lower(), 0) + 1
            entries.append([name, "R", rid, ("RRUFF 樣品 · " + loc)[:90] if loc else "RRUFF 樣品",
                            " ".join(sorted(els)), p])
    rruff_names = set(per_name)
    n_amcsd = {}

    # —— AMCSD
    za, zc = zipfile.ZipFile(z["dif.zip"]), zipfile.ZipFile(z["amc.zip"])
    amc = {n.rsplit("__", 1)[-1].split(".")[0]: n for n in zc.namelist()}
    groups = {}
    for n in za.namelist():
        code = n.rsplit("__", 1)[-1].split(".")[0]
        t = za.read(n).decode("latin-1")
        lines = t.splitlines()
        name = lines[0].strip()
        hdr = []
        for ln in lines[1:14]:
            if "_database_code" in ln or "CELL PARAMETERS" in ln:
                break
            hdr.append(ln.strip())
        amb, cond = condition(hdr[3:] + hdr[2:3])
        author = (hdr[0].split(",")[0].split()[0] if hdr and hdr[0] else "")
        year = re.search(r"\((\d{4})\)", hdr[1] if len(hdr) > 1 else "")
        note = "AMCSD · %s %s" % (author, year.group(1) if year else "")
        if cond:
            note += " · " + cond
        yr = int(year.group(1)) if year else 0
        groups.setdefault(name.lower(), []).append((not amb, -yr, code, name, note.strip(), t))
        n_amcsd[name.lower()] = n_amcsd.get(name.lower(), 0) + 1
    for key, lst in groups.items():
        lst.sort(key=lambda r: (r[0], r[1], r[2]))  # 常溫常壓優先，其次較新的結構
        ambient = [r for r in lst if not r[0]]
        want = 1 if key in rruff_names else 2
        pick = ambient[:want] if ambient else ([] if key in rruff_names else lst[:1])
        for nonamb, _yr, code, name, note, t in pick:
            els = set()
            if code in amc:
                on = False
                for ln in zc.read(amc[code]).decode("latin-1").splitlines():
                    if ln.startswith("atom"):
                        on = True
                        continue
                    if on and ln.strip():
                        e = elem_of(ln.split()[0])
                        if e:
                            els.add(e)
            p = compact(peaks_of(t))
            if p:
                entries.append([name, "A", code, note, " ".join(sorted(els)), p])

    # 常見度：RRUFF 天然樣品數、AMCSD 結構數（附在每筆資料最後）
    for e in entries:
        k = e[0].lower()
        e.append(n_rruff.get(k, 0))
        e.append(n_amcsd.get(k, 0))
    # RRUFF 樣品若缺元素資料，以同名礦物補上
    by = {}
    for e in entries:
        if e[4]:
            by.setdefault(e[0].lower(), set()).update(e[4].split())
    for e in entries:
        if not e[4]:
            e[4] = " ".join(sorted(by.get(e[0].lower(), [])))

    db = {"version": date.today().isoformat(),
          "source": "RRUFF Project (rruff.net): RRUFF powder DIF + AMCSD",
          "cite": ["Lafuente B, Downs R T, Yang H, Stone N (2015) The power of databases: the RRUFF project. "
                   "In: Highlights in Mineralogical Crystallography, pp 1-30.",
                   "Downs R T, Hall-Wallace M (2003) The American Mineralogist Crystal Structure Database. "
                   "American Mineralogist 88, 247-250."],
          "fields": ["name", "source R=RRUFF A=AMCSD", "id", "note", "elements", "d:I:hkl;...", "RRUFF samples", "AMCSD records"],
          "e": entries}
    raw = json.dumps(db, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(OUT, "wb", compresslevel=9) as f:
        f.write(raw)
    print("%d 筆（RRUFF %d、AMCSD %d），%d 種名稱 → %s（%.1f MB）" % (
        len(entries), sum(1 for e in entries if e[1] == "R"), sum(1 for e in entries if e[1] == "A"),
        len({e[0].lower() for e in entries}), OUT, os.path.getsize(OUT) / 1048576))


if __name__ == "__main__":
    main()
