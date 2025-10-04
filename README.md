# Staple-AI-DEMO

A simple Streamlit app that extracts line items from receipts/invoices using a hosted AI API via OpenRouter. Supports PNG/JPG upload and local file paths.

## Features
- Upload PNG/JPG or provide local image path
- Calls a Vision API via OpenRouter and normalizes line items
- Displays cleaned table and total, with CSV export
- Optional OCR fallback (Tesseract) and PDF-to-image conversion

## Requirements
- Python 3.10+
- `pip install -r requirements.txt`
- An `OPENROUTER_API_KEY` (set securely, do not commit to GitHub)

## Local Run
1. Create a virtual environment (optional):
   - Windows: `python -m venv .venv && .venv\\Scripts\\activate`
   - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your API key:
   - Option A: create `AAPI.env` in project root: `OPENROUTER_API_KEY=sk-or-...`
   - Option B: set env var: `setx OPENROUTER_API_KEY sk-or-...` (Windows) or `export OPENROUTER_API_KEY=sk-or-...` (macOS/Linux)
4. Run: `streamlit run STA.py`

## GitHub
- This repo includes `.gitignore` to prevent committing secrets. Make sure `AAPI.env` and `.env` are NOT committed.
- Push to GitHub:
  ```bash
  git init
  git add .
  git commit -m "Initial commit"
  git branch -M main
  git remote add origin https://github.com/<your-username>/Staple-AI-DEMO.git
  git push -u origin main
  ```

## Streamlit Cloud Deployment
1. Go to Streamlit Cloud and deploy this GitHub repo.
2. Set app secrets:
   - In the app settings → Secrets, add:
     ```
     OPENROUTER_API_KEY = sk-or-...
     ```
   - (Optional) `APP_URL = https://<your-app>.streamlit.app`
3. Run the app. Use the sidebar "Test API" to confirm connectivity.
4. Upload a PNG/JPG, click "Analyze Receipt".

## Keeping Secrets Safe
- Do NOT commit `AAPI.env`, `.env`, or any API key to GitHub.
- Use Streamlit Cloud Secrets for deployed apps.
- Locally, prefer environment variables or a non-committed `.env`/`AAPI.env`.

## Troubleshooting
- 401 Unauthorized ("User not found"):
  - Ensure your `OPENROUTER_API_KEY` is set in Secrets or environment.
  - Verify the key is active in your OpenRouter dashboard; rotate if needed.
- Empty AI response:
  - Try another image or model; ensure the uploaded file is valid.
- Parsing issues:
  - The app expects strict JSON from the API; it extracts fenced JSON blocks and normalizes line items.

## License
Internal demo. Do not publish keys or sensitive data.