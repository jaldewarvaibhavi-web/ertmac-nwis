# Deploying eRTMAC-NWIS (GitHub → Streamlit Community Cloud)

## Files that make cloud deployment work
| File | Purpose |
|---|---|
| `requirements.txt` (repo root) | Python packages installed by Streamlit Cloud |
| `packages.txt` (repo root) | Linux package `tesseract-ocr` so scanned PDFs can be OCR-read in the cloud |
| `.streamlit/config.toml` | light theme, upload limit |
| `src/bootstrap.py` | on first start: builds `data/nwis.db` if missing and retrains the models if they are missing or were saved with a different scikit-learn version |
| `.gitignore` | keeps `.env`, `.streamlit/secrets.toml` and `venv/` out of GitHub |

## 1. Push to GitHub (Windows, VS Code terminal, inside the `nwis` folder)
```bash
git init
git add .
git commit -m "eRTMAC-NWIS prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/ertmac-nwis.git
git push -u origin main
```
Create the empty repository on github.com first (no README / .gitignore / licence, so the push does not conflict).

## 2. Deploy on Streamlit Community Cloud
share.streamlit.io → sign in with GitHub → **Create app** → **Yup, I have an app**
* Repository: `<your-username>/ertmac-nwis` · Branch: `main`
* Main file path: **`app/dashboard.py`**
* App URL: e.g. `ertmac-nwis`
* **Advanced settings** → Python **3.11** (or 3.12) → Secrets (optional):
  ```toml
  ANTHROPIC_API_KEY = "sk-ant-..."
  ```
* **Deploy**. First start installs packages and may retrain the models (a few minutes).

## 3. Updating
Change code locally → `git add .` → `git commit -m "..."` → `git push`. The app redeploys automatically.

## Notes
* Free apps sleep after inactivity; open the URL a few minutes before a presentation.
* The cloud file system is temporary: 👍/👎 feedback written to SQLite is lost when the app restarts.
* Only the Streamlit dashboard runs on Streamlit Cloud. The FastAPI server needs a separate host, e.g.
  Render: build `pip install -r requirements.txt`, start `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
* The data is synthetic (plus one public FORGE log) — safe for a public repository. Never commit real
  OIL / eRTMAC data or API keys.
