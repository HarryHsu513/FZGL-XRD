/* XRD Studio — 背景執行緒（module worker）：在瀏覽器裡載入 Python (Pyodide)、numpy、matplotlib 與 xrd_core.py。
   所有運算都在使用者自己的電腦上進行，檔案不會上傳到任何伺服器。
   需要的檔案都和 index.html 放在同一層：xrd_core.py、fonts.zip、minerals.json.gz
  （舊版的 fonts/、data/ 資料夾也仍然可以使用）。 */
const APP_VERSION = "2.3";
const PYODIDE_VERSION = "314.0.7";
const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v" + PYODIDE_VERSION + "/full/";
const FONT_FILES = [
  "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf",
  "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf",
];

const progress = text => self.postMessage({type: "progress", text});
let py = null, api = null, dbReady = null;

/* 依序嘗試多個位置，回傳第一個成功的 ArrayBuffer；全部失敗時回傳 null 並記錄嘗試過的網址 */
async function fetchFirst(paths, tried) {
  for (const p of paths) {
    const url = new URL(p + "?v=" + APP_VERSION, self.location.href);
    try {
      const r = await fetch(url);
      if (r.ok) return await r.arrayBuffer();
      tried.push(`${url.pathname} → HTTP ${r.status}`);
    } catch (e) {
      tried.push(`${url.pathname} → ${e.message}`);
    }
  }
  return null;
}

async function loadFonts() {
  const tried = [];
  const zip = await fetchFirst(["fonts.zip"], tried);
  if (zip) {
    py.FS.writeFile("/app/fonts.zip", new Uint8Array(zip));
    await py.runPythonAsync("import zipfile\nzipfile.ZipFile('/app/fonts.zip').extractall('/app/fonts')");
    return FONT_FILES.length;
  }
  let n = 0;   // 舊版：fonts/ 資料夾內的個別檔案
  for (const f of FONT_FILES) {
    const buf = await fetchFirst(["fonts/" + f], []);
    if (buf) { py.FS.writeFile("/app/fonts/" + f, new Uint8Array(buf)); n++; }
  }
  return n;
}

async function boot() {
  progress("下載 Python 執行環境…");
  const {loadPyodide} = await import(PYODIDE_BASE + "pyodide.mjs");
  py = await loadPyodide({indexURL: PYODIDE_BASE});
  progress("載入 numpy 與 matplotlib…");
  const codeP = fetch(new URL("xrd_core.py?v=" + APP_VERSION, self.location.href)).then(r => {
    if (!r.ok) throw new Error(`找不到 xrd_core.py（HTTP ${r.status}），請確認它和 index.html 放在同一層`);
    return r.text();
  });
  await py.loadPackage(["numpy", "matplotlib"]);
  progress("載入字型與轉換程式…");
  py.FS.mkdirTree("/app/fonts");
  py.FS.mkdirTree("/app/data");
  const code = await codeP;
  const nFonts = await loadFonts();
  py.FS.writeFile("/app/xrd_core.py", code);
  await py.runPythonAsync("import sys\nsys.path.insert(0, '/app')\nimport xrd_core\nxrd_core.register_fonts('/app/fonts')\n"
                          + "xrd_core.MINDB_PATH = '/app/data/minerals.json.gz'");
  api = py.pyimport("xrd_core").api;
  const info = JSON.parse(api("info", "{}").get(0));
  info.fontsLoaded = nFonts;
  return info;
}

/* 礦物資料庫（約 2.7 MB）只在第一次使用「鑑定」時下載 */
function ensureDB() {
  if (!dbReady) {
    dbReady = (async () => {
      const tried = [];
      const buf = await fetchFirst(["minerals.json.gz", "data/minerals.json.gz"], tried);
      if (!buf) {
        throw new Error("找不到礦物資料庫 minerals.json.gz。請把它放在和 index.html 同一層（GitHub 上請確認它有被上傳）。"
                        + "已嘗試：" + tried.join("；"));
      }
      py.FS.writeFile("/app/data/minerals.json.gz", new Uint8Array(buf));
    })().catch(err => { dbReady = null; throw err; });
  }
  return dbReady;
}

const ready = boot();
ready.then(info => self.postMessage({type: "ready", info}))
     .catch(err => self.postMessage({type: "fatal", error: String((err && err.message) || err)}));

self.onmessage = async e => {
  const {id, cmd, args} = e.data;
  try {
    await ready;
    if (cmd.startsWith("id_")) await ensureDB();
    const res = api(cmd, JSON.stringify(args || {}));
    const meta = JSON.parse(res.get(0));
    let bytes = res.get(1);
    if (bytes && typeof bytes.toJs === "function") { const p = bytes; bytes = p.toJs(); p.destroy(); }
    res.destroy();
    if (bytes && !(bytes instanceof Uint8Array)) bytes = new Uint8Array(bytes);
    self.postMessage({type: "result", id, meta, bytes: bytes || null}, bytes ? [bytes.buffer] : []);
  } catch (err) {
    self.postMessage({type: "result", id, meta: {error: String((err && err.message) || err)}, bytes: null});
  }
};
