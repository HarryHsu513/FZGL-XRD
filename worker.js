/* XRD Studio — 背景執行緒（module worker）：在瀏覽器裡載入 Python (Pyodide)、numpy、matplotlib 與 xrd_core.py。
   所有運算都在使用者自己的電腦上進行，檔案不會上傳到任何伺服器。 */
const APP_VERSION = "2.0";
const PYODIDE_VERSION = "314.0.7";
const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v" + PYODIDE_VERSION + "/full/";
const FONT_FILES = [
  "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf",
  "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf",
];

const progress = text => self.postMessage({type: "progress", text});
let api = null;

async function boot() {
  progress("下載 Python 執行環境…");
  const {loadPyodide} = await import(PYODIDE_BASE + "pyodide.mjs");
  const py = await loadPyodide({indexURL: PYODIDE_BASE});
  progress("載入 numpy 與 matplotlib…");
  const files = Promise.all([
    fetch("xrd_core.py?v=" + APP_VERSION).then(r => { if (!r.ok) throw new Error("找不到 xrd_core.py"); return r.text(); }),
    ...FONT_FILES.map(f => fetch("fonts/" + f + "?v=" + APP_VERSION).then(r => (r.ok ? r.arrayBuffer() : null)).catch(() => null)),
  ]);
  await py.loadPackage(["numpy", "matplotlib"]);
  progress("載入字型與轉換程式…");
  const [code, ...fonts] = await files;
  py.FS.mkdirTree("/app/fonts");
  fonts.forEach((buf, i) => { if (buf) py.FS.writeFile("/app/fonts/" + FONT_FILES[i], new Uint8Array(buf)); });
  py.FS.writeFile("/app/xrd_core.py", code);
  await py.runPythonAsync("import sys\nsys.path.insert(0, '/app')\nimport xrd_core\nxrd_core.register_fonts('/app/fonts')");
  api = py.pyimport("xrd_core").api;
  const info = JSON.parse(api("info", "{}").get(0));
  info.fontsLoaded = fonts.filter(Boolean).length;
  return info;
}

const ready = boot();
ready.then(info => self.postMessage({type: "ready", info}))
     .catch(err => self.postMessage({type: "fatal", error: String((err && err.message) || err)}));

self.onmessage = async e => {
  const {id, cmd, args} = e.data;
  try {
    await ready;
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
