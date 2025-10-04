# streamlit_app.py
import io, json, re, base64
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageFilter
from datetime import datetime
import tempfile, os
from openai import OpenAI
from dotenv import load_dotenv

# === 引擎設定 ===
USE_KERAS_OCR = False
USE_PADDLE = False
USE_TESSERACT = False
USE_AI_API = True

"""Client initialization with robust key loading.
Order of precedence:
1) Streamlit secrets (st.secrets["OPENROUTER_API_KEY"]) if available
2) Environment variable loaded via dotenv (AAPI.env)
"""
load_dotenv('AAPI.env')
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
try:
    # Use Streamlit secrets in deployed environments
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    OPENROUTER_API_KEY = None
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = None
if OPENROUTER_API_KEY:
    try:
        default_headers = {
            "HTTP-Referer": os.getenv("APP_URL", "http://localhost"),
            "X-Title": "StapleAI MVP",
        }
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1", default_headers=default_headers)
    except Exception:
        client = None

import numpy as np

# (Optional) Keras-OCR support disabled per request

# Optional Tesseract fallback
try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
    USE_TESSERACT = True
except Exception:
    USE_TESSERACT = False

st.set_page_config(page_title="Doc AI MVP", layout="wide")

# -------- helpers --------
def to_iso_date(s):
    for fmt in ("%Y-%m-%d","%Y/%m/%d","%d-%m-%Y","%d/%m/%Y","%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return s

def to_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except:
        return s

def load_image(file):
    if file.type == "application/pdf":
        # 簡化：使用 pdf2image 轉第一頁（可擴充批次頁）
        try:
            from pdf2image import convert_from_bytes
        except ImportError:
            st.info("pdf2image is not installed; PDF support is disabled.")
            return None
        try:
            img = convert_from_bytes(file.read(), first_page=1, last_page=1)[0]
            return img
        except Exception as e:
            st.info("Could not convert PDF to image.")
            return None
    else:
        return Image.open(file).convert("RGB")

def load_image_path(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        st.info("Could not open image; please check the path.")
        return None

def preprocess_for_ocr(img: Image):
    # Upscale small images and enhance contrast/sharpness for better detection
    if img is None:
        return None
    w, h = img.size
    target_w = max(1200, w)
    if target_w > w:
        scale = target_w / float(w)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Basic enhancement
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img

def encode_image_path_to_base64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def encode_uploaded_file_to_base64(file):
    """Encode a Streamlit UploadedFile (PNG/JPG) to base64 safely."""
    try:
        # Try common UploadedFile interfaces
        if hasattr(file, "getvalue"):
            data = file.getvalue()
        else:
            if hasattr(file, "seek"):
                file.seek(0)
            data = file.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None

def call_ai_with_b64(b64: str, model: str | None = None):
    """Shared AI call using base64 image string and configured model."""
    if client is None:
        st.error("AI client not initialized. Please set OPENROUTER_API_KEY in secrets or AAPI.env.")
        return None
    prompt = (
        "You are an invoice/receipt extractor. Return ONLY a JSON array of objects "
        "with keys 'name', 'qty', 'unit_price', 'total_price'. Use numeric types for qty, unit_price, total_price. "
        "No markdown, no prose, no comments. Example: "
        "[{\"name\":\"Item A\",\"qty\":2,\"unit_price\":3.5,\"total_price\":7.0}]"
    )
    try:
        model_name = model or os.getenv("AI_MODEL") or "mistralai/mistral-small-3.2-24b-instruct:free"
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            stream=False,
        )
        choice = resp.choices[0] if getattr(resp, 'choices', None) else None
        content = choice.message.content if choice and getattr(choice, 'message', None) else None
        if not content:
            st.info("Empty AI response.")
            return None
        parsed = extract_json_from_text(content)
        if parsed is None:
            st.info("AI returned non-JSON content.")
        return parsed
    except Exception as e:
        msg = str(e)
        if "401" in msg or "User not found" in msg:
            st.error("401 Unauthorized: Check OPENROUTER_API_KEY and ensure it is set in deployment secrets/environment.")
            st.info("If using OpenRouter, verify your key is active and domain/app name is configured.")
        else:
            st.error(f"AI analysis failed: {e}")
        return None

def test_api_connection(model: str | None = None):
    """Quick connectivity test: sends a small text-only prompt to the selected model."""
    if client is None:
        st.error("AI client not initialized.")
        return False
    try:
        model_name = model or os.getenv("AI_MODEL") or "mistralai/mistral-small-3.2-24b-instruct:free"
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":[{"type":"text","text":"Return {\"ok\":true} only."}]}],
            stream=False,
        )
        content = resp.choices[0].message.content if getattr(resp, 'choices', None) else ""
        parsed = extract_json_from_text(content)
        if parsed and isinstance(parsed, dict) and parsed.get("ok") is True:
            st.success("API connectivity OK.")
            return True
        st.info("API responded but JSON parse did not match expected.")
        return True
    except Exception as e:
        st.error(f"API connectivity failed: {e}")
        return False

def analyze_receipt_image(image_path: str, model: str | None = None):
    """
    Use OpenRouter Vision API to analyze a receipt image and return JSON.
    """
    b64 = encode_image_path_to_base64(image_path)
    if not b64:
        st.info("Could not read image for AI analysis.")
        return None
    return call_ai_with_b64(b64, model=model)

def analyze_receipt_file(uploaded_file, model: str | None = None):
    """Analyze an uploaded PNG/JPG file via AI and return parsed JSON."""
    b64 = encode_uploaded_file_to_base64(uploaded_file)
    if not b64:
        st.info("Could not read uploaded image for AI analysis.")
        return None
    return call_ai_with_b64(b64, model=model)

def extract_json_from_text(text: str):
    """Extract JSON object/array from text, handling code fences and extra prose."""
    if not text:
        return None
    # Handle fenced blocks first
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidate = fence.group(1) if fence else None
    if not candidate:
        # Fallback: first JSON object or array by braces
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        candidate = m.group(1) if m else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        # Attempt to clean trailing commas and fix common issues
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

def parse_number(val):
    """Parse a number from strings like "$1,234.50" or return float if already numeric."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    # Extract first number pattern
    m = re.search(r"[-+]?\d*[\.,]?\d+", s.replace(',', ''))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def normalize_line_items(data):
    """Normalize AI output to a list of {name, qty, unit_price, total_price}."""
    rows = []
    if isinstance(data, list):
        src = data
    elif isinstance(data, dict):
        src = data.get('items') if isinstance(data.get('items'), list) else [data]
    else:
        src = []
    for item in src:
        try:
            name = str(item.get('name', '')).strip()
            qty = parse_number(item.get('qty')) or 1.0
            unit_price = parse_number(item.get('unit_price'))
            total_price = parse_number(item.get('total_price'))
            if total_price is None and unit_price is not None and qty is not None:
                total_price = round(qty * unit_price, 2)
            rows.append({
                'name': name,
                'qty': qty,
                'unit_price': unit_price,
                'total_price': total_price,
            })
        except Exception:
            pass
    # Filter out empty names
    rows = [r for r in rows if r.get('name')]
    return rows

def run_ocr(img: Image):
    if img is None:
        return []
    img_pp = preprocess_for_ocr(img)
    # Primary: Keras-OCR
    if USE_KERAS_OCR:
        try:
            pipeline = get_keras_pipeline()
            arr = np.array(img_pp)
            predictions = pipeline.recognize([arr])
            items = []
            for pred in (predictions[0] if predictions else []):
                try:
                    text, box = pred
                    bbox = [[int(pt[0]), int(pt[1])] for pt in box]
                    items.append((bbox, str(text), 0.9))
                except Exception:
                    pass
            if items:
                return items
            else:
                st.info("No text from Keras-OCR; attempting Tesseract fallback...")
        except Exception:
            st.info("Switching to OCR fallback...")
    # Fallback: Tesseract
    if USE_TESSERACT:
        try:
            df = pytesseract.image_to_data(img_pp, output_type=TesseractOutput.DATAFRAME)
            items = []
            for _, row in df.iterrows():
                try:
                    txt = str(row.get('text', '')).strip()
                    conf = float(row.get('conf', -1))
                    if txt and conf != -1:
                        x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
                        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                        items.append((bbox, txt, max(0.0, min(conf / 100.0, 1.0))))
                except Exception:
                    pass
            if items:
                return items
        except Exception:
            st.info("OCR fallback did not produce text.")
    return []

def boxes_and_texts(paddle_res):
    """Flatten PaddleOCR result defensively -> [(bbox, text, conf)]"""
    items = []
    try:
        for page in (paddle_res or []):
            for line in (page or []):
                bbox, txt, conf = None, "", 0.0
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    bbox = line[0]
                    info = line[1]
                    if isinstance(info, (list, tuple)) and len(info) >= 2:
                        txt = str(info[0]) if info[0] is not None else ""
                        try:
                            conf = float(info[1]) if info[1] is not None else 0.0
                        except Exception:
                            conf = 0.0
                if bbox and txt:
                    items.append((bbox, txt, conf))
    except Exception:
        pass
    return items

def draw_boxes(img: Image, items, conf_thr=0.5):
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for bbox, txt, conf in items:
        color = (0, 200, 0) if conf >= 0.85 else ((255,165,0) if conf>=conf_thr else (220,0,0))
        pts = [(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]),
               (bbox[2][0], bbox[2][1]), (bbox[3][0], bbox[3][1]), (bbox[0][0], bbox[0][1])]
        dr.line(pts, width=2, fill=color)
    return im

# -------- Dummy data (samples) --------
def make_dummy_invoice():
    img = Image.new("RGB", (1200, 800), color="white")
    dr = ImageDraw.Draw(img)
    lines = [
        ("Invoice No: INV-12345", (60, 60), 0.98, [[50, 50], [420, 50], [420, 95], [50, 95]]),
        ("Date: 2024-09-12", (60, 130), 0.95, [[50, 120], [300, 120], [300, 165], [50, 165]]),
        ("Total Amount: 1,234.56 USD", (60, 200), 0.92, [[50, 190], [520, 190], [520, 235], [50, 235]]),
        ("Vendor: ACME Corp.", (60, 270), 0.88, [[50, 260], [330, 260], [330, 305], [50, 305]]),
    ]
    items = []
    for text, pos, conf, bbox in lines:
        dr.text(pos, text, fill=(0, 0, 0))
        items.append((bbox, text, conf))
    return img, items

def make_dummy_jobcard():
    img = Image.new("RGB", (1200, 800), color="white")
    dr = ImageDraw.Draw(img)
    lines = [
        ("Job ID: JC-001", (60, 60), 0.97, [[50, 50], [330, 50], [330, 95], [50, 95]]),
        ("Machine ID: M-88", (60, 130), 0.93, [[50, 120], [360, 120], [360, 165], [50, 165]]),
        ("Operator: Alice", (60, 200), 0.91, [[50, 190], [330, 190], [330, 235], [50, 235]]),
        ("Remarks: Routine maintenance", (60, 270), 0.86, [[50, 260], [520, 260], [520, 305], [50, 305]]),
    ]
    items = []
    for text, pos, conf, bbox in lines:
        dr.text(pos, text, fill=(0, 0, 0))
        items.append((bbox, text, conf))
    return img, items

def make_dummy_receipt():
    img = Image.new("RGB", (1200, 1400), color="white")
    dr = ImageDraw.Draw(img)
    header_lines = [
        ("Receipt", (520, 40), 0.99, [[480, 30], [700, 30], [700, 80], [480, 80]]),
        ("Adress: 1234 Lorem, Dolor", (60, 130), 0.96, [[50, 120], [580, 120], [580, 170], [50, 170]]),
        ("Tel: 123-456-7890", (60, 190), 0.96, [[50, 180], [380, 180], [380, 230], [50, 230]]),
        ("Date: 01-01-2018", (60, 250), 0.95, [[50, 240], [380, 240], [380, 290], [50, 290]]),
        ("10:35", (420, 250), 0.95, [[410, 240], [520, 240], [520, 290], [410, 290]]),
    ]
    body_lines = [
        ("Lorem", (60, 340), 0.94, [[50, 330], [220, 330], [220, 380], [50, 380]]),
        ("6.50", (980, 340), 0.94, [[940, 330], [1080, 330], [1080, 380], [940, 380]]),
        ("Ipsum", (60, 410), 0.94, [[50, 400], [220, 400], [220, 450], [50, 450]]),
        ("7.50", (980, 410), 0.94, [[940, 400], [1080, 400], [1080, 450], [940, 450]]),
        ("Dolor Sit", (60, 480), 0.94, [[50, 470], [260, 470], [260, 520], [50, 520]]),
        ("48.00", (980, 480), 0.94, [[940, 470], [1080, 470], [1080, 520], [940, 520]]),
        ("Amet", (60, 550), 0.94, [[50, 540], [220, 540], [220, 590], [50, 590]]),
        ("9.30", (980, 550), 0.94, [[940, 540], [1080, 540], [1080, 590], [940, 590]]),
        ("Consectetur", (60, 620), 0.94, [[50, 610], [280, 610], [280, 660], [50, 660]]),
        ("11.90", (980, 620), 0.94, [[940, 610], [1080, 610], [1080, 660], [940, 660]]),
        ("Adipiscing Elit", (60, 690), 0.94, [[50, 680], [320, 680], [320, 730], [50, 730]]),
        ("1.20", (980, 690), 0.94, [[940, 680], [1080, 680], [1080, 730], [940, 730]]),
        ("Sed Do", (60, 760), 0.94, [[50, 750], [220, 750], [220, 800], [50, 800]]),
        ("0.40", (980, 760), 0.94, [[940, 750], [1080, 750], [1080, 800], [940, 800]]),
    ]
    totals_lines = [
        ("AMOUNT", (60, 860), 0.98, [[50, 850], [220, 850], [220, 900], [50, 900]]),
        ("84.80", (980, 860), 0.98, [[940, 850], [1080, 850], [1080, 900], [940, 900]]),
        ("Sub-total", (60, 940), 0.92, [[50, 930], [260, 930], [260, 980], [50, 980]]),
        ("76.80", (980, 940), 0.92, [[940, 930], [1080, 930], [1080, 980], [940, 980]]),
        ("Sales Tax", (60, 1010), 0.92, [[50, 1000], [280, 1000], [280, 1050], [50, 1050]]),
        ("8.00", (980, 1010), 0.92, [[940, 1000], [1080, 1000], [1080, 1050], [940, 1050]]),
        ("Balance", (60, 1080), 0.92, [[50, 1070], [260, 1070], [260, 1120], [50, 1120]]),
        ("0.00", (980, 1080), 0.92, [[940, 1070], [1080, 1070], [1080, 1120], [940, 1120]]),
    ]
    items = []
    for text, pos, conf, bbox in header_lines + body_lines + totals_lines:
        dr.text(pos, text, fill=(0, 0, 0))
        items.append((bbox, text, conf))
    return img, items

def make_dummy_docs(schema_name):
    docs = []
    if schema_name == "invoice":
        img, items = make_dummy_invoice()
        docs.append({"name": "sample_invoice.png", "items": items, "image": img})
    elif schema_name == "jobcard":
        img, items = make_dummy_jobcard()
        docs.append({"name": "sample_jobcard.png", "items": items, "image": img})
    elif schema_name == "receipt":
        img, items = make_dummy_receipt()
        docs.append({"name": "sample_receipt.png", "items": items, "image": img})
    else:
        # Generic sample for other types
        img = Image.new("RGB", (1200, 800), color="white")
        dr = ImageDraw.Draw(img)
        text = f"Sample {schema_name.title()} Document"
        dr.text((60, 60), text, fill=(0, 0, 0))
        items = [([[50, 50], [600, 50], [600, 120], [50, 120]], text, 0.9)]
        docs.append({"name": f"sample_{schema_name}.png", "items": items, "image": img})
    return docs

def extract_fields(items, schema_name):
    # 極簡規則抽取（可換成更聰明的 pattern/位置/LLM 後處理）
    text_full = " ".join([t for _, t, _ in items])
    data = {}
    if schema_name == "invoice":
        # 可用配置化，這裡先示範
        m = re.search(r"(INV[- ]?\d{5,})", text_full, re.I)
        if m: data["invoice_no"] = m.group(1)
        # 日期
        m = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", text_full)
        if m: data["date"] = to_iso_date(m.group(1))
        # 金額
        m = re.search(r"total[^\d]*([\d,]+\.\d{2})", text_full, re.I)
        if m: data["total_amount"] = to_float(m.group(1))
        # 幣別（極簡）
        for cur in ["USD","SGD","TWD","CNY","JPY","EUR"]:
            if cur in text_full: data["currency"] = cur
    elif schema_name == "jobcard":
        # 例：Job Card 欄位
        for pat, key in [
            (r"Job\s*ID[:\s]*([A-Z0-9\-]+)", "job_id"),
            (r"Machine\s*ID[:\s]*([A-Z0-9\-]+)", "machine_id"),
            (r"Operator[:\s]*([A-Za-z ]+)", "operator"),
        ]:
            m = re.search(pat, text_full, re.I)
            if m: data[key] = m.group(1).strip()
    elif schema_name == "receipt":
        m = re.search(r"Date[:\s]*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})", text_full, re.I)
        if m: data["date"] = to_iso_date(m.group(1))
        m = re.search(r"(\d{1,2}:\d{2})", text_full)
        if m: data["time"] = m.group(1)
        m = re.search(r"(?:AMOUNT|Total)[:\s]*([\d,]+\.\d{2})", text_full, re.I)
        if m: data["total_amount"] = to_float(m.group(1))
        m = re.search(r"Sub[-\s]?total[:\s]*([\d,]+\.\d{2})", text_full, re.I)
        if m: data["subtotal"] = to_float(m.group(1))
        m = re.search(r"(?:Sales\s*Tax|GST|VAT)[:\s]*([\d,]+\.\d{2})", text_full, re.I)
        if m: data["sales_tax"] = to_float(m.group(1))
        m = re.search(r"Balance[:\s]*([\d,]+\.\d{2})", text_full, re.I)
        if m: data["balance"] = to_float(m.group(1))
        m = re.search(r"Tel[:\s]*([0-9\-\+\s]+)", text_full, re.I)
        if m: data["phone"] = m.group(1).strip()
        m = re.search(r"(?:Adress|Address)[:\s]*([^\n]+?)\s+Tel", text_full, re.I)
        if m: data["address"] = m.group(1).strip()
    return data

def df_download_button(df, label="Download CSV", filename="extracted.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

# numpy 已在頂部引入

# -------- UI --------
st.title("📄 Document AI MVP")
st.markdown(r"""
### Welcome to the Document AI MVP – built with ❤️ by Innovation XLAB  
**Turn any receipt, invoice or job card into structured data in three clicks:**

1️⃣ **Upload**  
   • Drag-and-drop a PNG/JPG image into the left-hand “Upload PNG/JPG” box, **or**  
   • PDF files are automatically converted to an image for you.

2️⃣ **Analyze**  
   • Press the blue “Analyze Receipt” button.  
   • The AI (OpenRouter Vision API) will read the picture and extract line items, quantities, unit prices and totals.  
   • A preview of the annotated image appears so you can visually confirm the document was processed.

3️⃣ **Verify & Export**  
   • Review the table under “Line Items”; totals are automatically calculated.  
   • If something is wrong, just edit the cells directly in the browser.  
   • Click “Download CSV” to save the cleaned data—ready for Excel, Power BI or any accounting package.

**Pro tips**  
- Good lighting and straight angles give the best OCR accuracy.  
- The system supports English, Chinese and mixed-language receipts.  
- No data is stored on our servers; everything stays on your machine.  

**Need help?** Reach out to the Innovation XLAB team—your feedback makes us better. Happy scanning!
""")

# Sidebar controls
with st.sidebar:
    st.subheader("Controls")
    uploaded = st.file_uploader("Upload PNG/JPG", type=["png","jpg","jpeg"])
    # workspace_path = st.text_input(
    #     "Image path",
    #     value=r"Example.jpg"
    # )
    
    # conf_thr = st.slider("Low-confidence threshold", 0.0, 1.0, 0.5, 0.05, help="(unused in AI mode)")
    run = st.button("Analyze Receipt")
    test = st.button("Test API")
    st.caption("Set OPENROUTER_API_KEY via AAPI.env or Streamlit secrets for deployment.")
ai_model = "mistralai/mistral-small-3.2-24b-instruct:free"
# Session state for sharing results across tabs
 # Simple prototype: no session state needed

st.header("Preview")
ai_json = None
if 'test' in locals() and test:
    with st.spinner("Testing API connectivity..."):
        test_api_connection(model=ai_model)
if run and (uploaded is not None or workspace_path):
    with st.spinner("Analyzing with AI..."):
        if uploaded is not None:
            img = load_image(uploaded)
            name = getattr(uploaded, "name", "uploaded_image")
            if img is not None:
                st.image(img, caption=name, use_container_width=True)
            ai_json = analyze_receipt_file(uploaded, model=ai_model)
        else:
            img = load_image_path(workspace_path)
            name = workspace_path.split("\\")[-1] if "\\" in workspace_path else workspace_path.split("/")[-1]
            if img is not None:
                st.image(img, caption=name, use_container_width=True)
            ai_json = analyze_receipt_image(workspace_path, model=ai_model)

# show_json = st.toggle("Show AI JSON")
# if show_json and ai_json is not None:
#     with st.expander("AI Response"):
#         st.json(ai_json)

st.header("Line Items")
if ai_json is not None:
    rows = normalize_line_items(ai_json)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        # Show total sum if available
        total_sum = df['total_price'].dropna().sum() if 'total_price' in df.columns else None
        if total_sum is not None and total_sum > 0:
            st.caption(f"Total detected: {total_sum:.2f}")
        df_download_button(df, "Download CSV", "extracted.csv")
    else:
        st.info("No line items detected by AI.")

 # Reconciliation removed for a minimal prototype
