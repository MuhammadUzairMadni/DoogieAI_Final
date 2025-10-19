# ==============================================================
# Doogie AI - PDF Processing API
# ==============================================================
# Features:
# - PDF upload & text extraction
# - Multi-language detection & translation
# - Medical reasoning using official Doogie Master Prompt
# - Reference extraction (NICE, Oxford, NHS)
# - NICE Care Pathway bundle builder
# ==============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from docx import Document
from pydantic import BaseModel
import shutil
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import json
from dotenv import load_dotenv
import openai

# ------------------- Load .env -------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

openai.api_key = OPENAI_API_KEY

# ------------------- Load Doogie Master Prompt -------------------
def load_master_prompt(docx_path: str) -> str:
    """Load and combine all paragraphs from the Doogie Master Prompt Word file."""
    try:
        doc = Document(docx_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return full_text
    except Exception as e:
        raise RuntimeError(f"❌ Error loading master prompt: {str(e)}")

MASTER_PROMPT_PATH = "doogie_master_prompt.docx"
MASTER_PROMPT_TEXT = load_master_prompt(MASTER_PROMPT_PATH)
print("✅ Doogie Master Prompt loaded successfully!")

# ------------------- FastAPI Setup -------------------
app = FastAPI(title="Doogie AI - PDF Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ------------------- Upload Endpoint -------------------
@app.post("/upload-pdf/")
def upload_pdf(file: UploadFile = File(...)):
    """Upload and save a PDF to the server."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed!")

    file_path = UPLOAD_DIR / Path(file.filename).name
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "✅ PDF uploaded successfully!", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ------------------- Extract PDF Metadata -------------------
def extract_pdf_info(pdf_path: Path) -> dict:
    info = {}
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            doc_info = reader.metadata
            if doc_info:
                info = {k: str(v) for k, v in doc_info.items()}
            info["num_pages"] = len(reader.pages)
    except Exception as e:
        info["error"] = str(e)
    return info

# ------------------- Extract PDF Text -------------------
def extract_pdf_text(pdf_path: Path) -> str:
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        text = f"Error extracting text: {str(e)}"
    return text

# ------------------- Extract Images -------------------
def extract_pdf_images(pdf_path: Path):
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            for img_index, img in enumerate(doc[page_index].get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                image = Image.open(io.BytesIO(image_data))
                images.append({
                    "id": f"page{page_index}_img{img_index}",
                    "width": image.width,
                    "height": image.height
                })
        doc.close()
    except Exception as e:
        print("⚠️ Image extraction error:", e)
    return images

# ------------------- Language Detection -------------------
def detect_language(text: str) -> str:
    if not text.strip():
        return "No text to detect"
    
    prompt = f"Detect the language of the following text and respond with only the language name:\n\n{text[:1000]}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error detecting language: {str(e)}"

# ------------------- Translation -------------------
def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang.lower() == "english":
        return text

    prompt = f"Translate this {source_lang} medical report into clear English, preserving medical meaning:\n\n{text[:3000]}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional medical text translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error translating text: {str(e)}"

# ------------------- Analysis (Doogie Master Prompt) -------------------
def analyze_text_with_doogie(text: str) -> str:
    """Use the Master Prompt for structured medical reasoning analysis."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": MASTER_PROMPT_TEXT},
                {"role": "user", "content": f"Analyze this patient report:\n\n{text}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing text: {str(e)}"

# ------------------- Reference Extraction -------------------
class ReferenceRequest(BaseModel):
    text: str

@app.post("/extract-references/")
def extract_references(request: ReferenceRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for reference extraction")

    prompt = f"""
    You are Doogie AI's biomedical reference extraction engine.
    Extract Oxford, NICE, NHS, or journal references and return valid JSON array.
    Format:
    [
      {{"source": "<reference>", "type": "<Book/NICE/Journal/NHS>", "citation": "<short id>"}}
    ]
    Text: {text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract and validate medical references."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return {"references": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting references: {str(e)}")

# ------------------- Read + Analyze PDF -------------------
@app.get("/read-pdf/{filename}")
def read_pdf_file(filename: str, extract_text: bool = Query(True), extract_images: bool = Query(False)):
    """Read, translate (if needed), analyze PDF medical report, and extract references."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    result = {
        "filename": filename,
        "metadata": extract_pdf_info(file_path),
    }

    if extract_text:
        text = extract_pdf_text(file_path)
        language = detect_language(text)
        translated = translate_to_english(text, language)
        analysis = analyze_text_with_doogie(translated)
        refs = extract_references(ReferenceRequest(text=translated))

        result.update({
            "language_detected": language,
            "translated_text": translated,
            "analysis": analysis,
            "references": refs,
        })

    if extract_images:
        result["images"] = extract_pdf_images(file_path)

    return JSONResponse(content=result)

# ------------------- NICE Care Pathway Bundle Builder -------------------
from care_pathways.bundle_builder import build_bundle

class PathwayBuildRequest(BaseModel):
    condition: str

@app.post("/build-pathway-bundle/")
def build_pathway_bundle(request: PathwayBuildRequest):
    """Build a NICE Care Pathway Bundle for a specific condition."""
    try:
        output_path = build_bundle(request.condition)
        return {
            "message": f"✅ {request.condition.capitalize()} pathway bundle built successfully!",
            "bundle_path": str(output_path)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building bundle: {str(e)}")

# ------------------- Root -------------------
@app.get("/")
def root():
    return {"message": "🚀 Doogie AI PDF API running with Translation, Analysis, and Master Prompt Integration!"}

# ------------------- Run -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
