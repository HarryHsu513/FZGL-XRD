# XRD Studio

同步輻射（.xye / .xyn）與 In-house（.txt / .xy / .dat / .csv）XRD 數據的波長轉換與期刊級繪圖工具。
打開網頁就能用：自動判斷格式、把 2θ 換算到 In-house 波長（Cu Kα₁ 1.540598 Å）、即時預覽、匯出 PNG / TIFF / PDF / SVG / EPS 與 CSV。

所有運算都在使用者自己的瀏覽器裡用 Python（Pyodide + numpy + matplotlib）執行，**檔案不會上傳到任何伺服器**。

## 資料夾內容

所有檔案都放在同一層（沒有子資料夾，GitHub 網頁上傳時才不會漏掉）：

| 檔案 | 用途 |
|---|---|
| `index.html` | 網頁介面 |
| `worker.js` | 在瀏覽器背景載入 Python 執行環境 |
| `xrd_core.py` | 讀檔、格式判斷、波長換算（保留原 `convert_two_theta()`）、繪圖、匯出與礦物鑑定 |
| `fonts.zip` | Liberation Sans / Serif 字型（與 Arial / Times New Roman 字寬完全相同，SIL OFL 授權）；不用解壓縮 |
| `minerals.json.gz` | 礦物參考峰表（RRUFF + AMCSD，12,894 筆、9,121 種）；不用解壓縮 |
| `run_local.py` | 在自己電腦執行（並代為查詢 webmineral）；缺少 `minerals.json.gz` 時會自動下載建立 |
| `tools/build_mineral_db.py` | （選用）重新下載 RRUFF 資料並更新 `minerals.json.gz`，架站時不需要上傳 |

開啟時若缺少 `fonts.zip` 或 `minerals.json.gz`，畫面上方會直接提示缺了哪個檔案。

## 礦物鑑定（「鑑定」分頁）

有兩種方式，可以互相對照：

**自動全譜辨識（建議先用）**
用 RRUFF / AMCSD 的每一筆參考資料和整條圖譜比對，而不只是看幾根峰：
1. 全資料庫篩選：所有參考資料都和偵測到的峰比對，容許 2θ 位移與 ±1.2% 晶格縮放（天然礦物的固溶體）。
2. 全譜擬合：每個候選礦物以峰表（實驗室數據含 Kα2）產生理論圖譜，逐步加入最能降低擬合殘差的礦物，並精修位移、晶格與峰寬。
3. 分級：「確定」＝明顯改善擬合、強峰都在、且有多根只能由它解釋的峰；「可能」＝證據較弱但不排除（靈敏度選「高」時會盡量列出）。罕見礦物需要更強的證據。
4. 每個礦物列出圖譜幾乎相同的替代礦物（例如同結構、不同成分），需依化學組成判斷。
結果附擬合圖（觀測、擬合、殘差、各礦物峰位），可一鍵把所有礦物加到圖上或標上符號。

準確率測試：用 RRUFF 的實測純礦物粉末圖譜，在電腦中混成 60 個已知成分的樣品（206 個礦物相，含 25 個 < 5% 的少量相，並扣除該樣品本身的參考資料），在設定可能元素的情況下：

| | 逐峰反查 | 自動全譜辨識 |
|---|---|---|
| 找回率（名稱完全相同） | 70.9% | 76.7% |
| 找回率（含替代礦物 / 同固溶體系列） | 75.7% | 87.4% |
| 列出礦物中正確的比例 | 31.8% | 52.1% |
| 每個樣品的錯誤礦物數 | 5.2 | 2.4 |
| < 5% 的少量相 | 4/25 | 11–13/25 |
| 「確定」級的正確比例 | — | 83% |

**逐峰反查（學長的流程）**
1. **最強峰**：自動偵測峰位，勾選 1–3 根作為 D1–D3。
2. **webmineral 反查**：以 d 值查詢 webmineral 的 X-Ray 資料表；可設定誤差與可能含有的元素。開啟「不限峰的強弱順序」時，每根峰可對應礦物三強峰中的任一根。
3. **RRUFF 驗證**：候選礦物以完整峰表比對並排序，可展開看每根參考峰對到哪根觀測峰。
4. **確認後再查**：確認後它能解釋的峰會標為已解釋，自動選取剩下最強的峰再查。

webmineral 不允許其他網頁從瀏覽器直接讀取查詢結果，所以要在工具內看到 webmineral 的候選，請用 `python run_local.py` 執行（由本機代為查詢）。放在 GitHub Pages 上時，會改用 RRUFF 資料庫依同樣的三強峰規則篩選，並提供「在 webmineral 開啟」按鈕對照。

吻合度只代表峰位圖譜的相似程度，最後仍需依樣品來源與化學組成判斷（例如石英與 berlinite 圖譜幾乎相同，只能靠元素區分）。

更新參考資料：`python tools/build_mineral_db.py`（會從 rruff.net 下載約 60 MB 的原始檔，並覆蓋同一層的 `minerals.json.gz`）。
資料引用：Lafuente B, Downs R T, Yang H, Stone N (2015) The power of databases: the RRUFF project; Downs R T, Hall-Wallace M (2003) American Mineralogist 88, 247–250。

## 免費架設到網路上（GitHub Pages）

GitHub Pages 是 GitHub 提供的免費靜態網站空間：不會休眠、不需維護主機，公開儲存庫（public repository）可免費使用。
本工具只有約 5 MB，遠低於 1 GB 的網站容量上限；每月 100 GB 流量也遠超過一般實驗室的使用量。

1. 到 <https://github.com> 註冊並登入。
2. 右上角 **＋ → New repository**。
   - Repository name 填 `xrd-studio`（這會成為網址的一部分）
   - 選 **Public**
   - 按 **Create repository**
3. 在新儲存庫的頁面點 **uploading an existing file**（或 **Add file → Upload files**）。
4. 把本資料夾**裡面的** 7 個檔案（`index.html`、`worker.js`、`xrd_core.py`、`fonts.zip`、`minerals.json.gz`、`run_local.py`、`README.md`）拖進上傳區，等每個檔案都出現在清單裡，再按 **Commit changes**。
   - ⚠️ `index.html` 必須在儲存庫的最上層，不要把外層資料夾整個拖進去。
   - ⚠️ `fonts.zip`、`minerals.json.gz` 保持壓縮檔原樣上傳，不要解壓縮。
   - 上傳後，儲存庫首頁的檔案清單應該看得到這 7 個檔案；少了 `minerals.json.gz`，「鑑定」功能就無法使用。
5. 到 **Settings → Pages**（左側選單「Code, planning, and automation」區）。
6. 在 **Build and deployment → Source** 選 **Deploy from a branch**；Branch 選 **main**、資料夾選 **/ (root)**，按 **Save**。
7. 等 1–2 分鐘後重新整理該頁，上方會出現網址：
   `https://你的帳號.github.io/xrd-studio/`
   把這個網址分享給別人，對方用瀏覽器打開就能直接使用。

### 之後要更新程式

到儲存庫頁面 **Add file → Upload files**，拖入修改過的檔案（同名會直接覆蓋）→ **Commit changes**，約 1 分鐘後網站就會更新。
使用者那端若仍看到舊版，重新整理頁面即可（瀏覽器最多快取約 10 分鐘）。

## 在自己電腦執行（自己使用時建議）

```bash
python run_local.py
```
會自動開啟 `http://127.0.0.1:8000/`，「鑑定」分頁可直接顯示 webmineral 的查詢結果。不能直接雙擊 `index.html`（瀏覽器安全限制不允許從本機檔案啟動背景執行緒）。

## 不開介面的批次轉換

輸出與原 `Synchrotron_Convert.py` 完全相同的 CSV（在原檔旁產生）：

```bash
python xrd_core.py 檔案1.xye 檔案2.xye
```

## 使用須知

- 第一次開啟需下載 Python 執行環境（約 20–30 MB，10–30 秒），之後由瀏覽器快取，再次開啟只需數秒。需要網路連線。
- 請使用最新版 Chrome、Edge、Safari 或 Firefox。
- 網頁版以 Liberation Sans 呈現 Arial、以 Liberation Serif 呈現 Times New Roman，兩者字寬與原字型完全相同，版面一致；SVG 匯出時會指定優先使用 Arial / Times New Roman。
- Nature 的正式稿主圖要求向量格式（PDF / EPS / AI / SVG），不接受 PNG / TIFF；投稿前請以各期刊最新作者須知為準。
- Python 執行環境由 jsDelivr CDN 提供（Pyodide 314.0.7，版本已固定），網站本身不需要任何後端主機。
