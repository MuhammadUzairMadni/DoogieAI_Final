from fastapi import FastAPI, UploadFile, File, HTTPException, Query,Form,Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import io
import os
from dotenv import load_dotenv
import openai
from pydantic import BaseModel
import json
from openai import AsyncOpenAI

# ------------------- Load .env -------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")
openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

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
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and save a PDF to the server."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed!")

    file_path = UPLOAD_DIR / Path(file.filename).name
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "✅ PDF uploaded successfully!", "filename": file.filename, "file_path": str(file_path)}
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

    prompt = (
        f"Translate the following {source_lang} medical text into clear English. "
        f"Keep medical meaning and context accurate:\n\n{text[:3000]}"
    )
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

# ------------------- Analysis -------------------
def analyze_text_with_doogie(text: str) -> str:
    prompt = f"""
    You are Doogie, an AI clinical assistant.
    Analyze the following medical report and extract structured details.
    Return JSON only with these fields:
    - patient_name
    - age
    - gender
    - diagnosis
    - symptoms
    - prescriptions
    - follow_up
    - red_flags
    - summary (2-3 sentences)

    Report:
    {text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a clinical reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing text: {str(e)}"

# ------------------- Step 2: Reference Model -------------------
class ReferenceRequest(BaseModel):
    text: str

# ------------------- STEP 2: Extract Oxford/NICE References -------------------
@app.post("/extract-references/")
async def extract_references(request: ReferenceRequest):
    """
    Extract Oxford, NICE, NHS, or Journal-style references from the given text.
    Input Example (Swagger body):
    {
        "text": "Refer to NICE Guideline NG136 and BMJ 2021;372:n85 for more info."
    }
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for reference extraction")

    prompt = f"""
    You are Doogie AI's reference extraction engine.
    From the following clinical or academic text, extract all identifiable references:
    - NICE Guidelines (e.g. NG136, NG245)
    - Oxford Handbook / medical books
    - BMJ / The Lancet / Journal citations
    - NHS documents or reports

    Return valid JSON in this exact structure:
    [
      {{
        "source": "<original reference text>",
        "type": "<Book / NICE Guideline / Journal / NHS Report / Unknown>",
        "citation": "<short name or ID if found>"
      }}
    ]

    Text:
    {text}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a biomedical reference extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        references = response.choices[0].message.content.strip()
        return JSONResponse(content={"references": references})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting references: {str(e)}")

# ------------------- Read + Analyze PDF -------------------
@app.get("/read-pdf/{filename}")
async def read_pdf_file(filename: str, extract_text: bool = Query(True), extract_images: bool = Query(False)):
    """Read, translate (if needed), analyze PDF medical report, and extract references."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    result = {
        "filename": filename,
        "file_path": str(file_path),
        "file_size": file_path.stat().st_size,
        "metadata": extract_pdf_info(file_path)
    }

    # --- Extract text ---
    if extract_text:
        text = extract_pdf_text(file_path)
        language = detect_language(text)
        result["text_content"] = text
        result["char_count"] = len(text)
        result["word_count"] = len(text.split())
        result["language_detected"] = language

        # --- Translate if needed ---
        translated_text = text if language.lower() == "english" else translate_to_english(text, language)
        result["translated_text"] = translated_text

        # --- Analyze automatically ---
        result["analysis"] = analyze_text_with_doogie(translated_text)

        # --- Extract references automatically ---
        try:
            ref_response = await extract_references(ReferenceRequest(text=translated_text))
            result["references"] = json.loads(ref_response.body.decode())
        except Exception as e:
            result["references"] = f"Error extracting references: {str(e)}"

    # --- Extract images (optional) ---
    if extract_images:
        result["images"] = extract_pdf_images(file_path)

    return JSONResponse(content=result)

DOOGIE_PROMPT = """
You are Doogie, an advanced AI clinical assistant.
Follow strict medical reasoning and SNOMED/FHIR-based structure.
Perform complete patient case analysis using provided details.
Always include structured reasoning, risk factors, and relevant clinical guidelines.
Respond with concise, medically accurate, and professional output.
"""
# ------------------- Unified Upload + Analyze -------------------
@app.post("/upload-and-analyze/")
async def upload_and_analyze(patients: list[dict] = Body(...)):
    """
    Analyze patient JSON data using Doogie system prompt.
    Works with synchronous OpenAI client.
    """
    results = []

    for patient in patients:
        user_input = (
            f"Patient Data:\n"
            f"Name: {patient.get('name')}\n"
            f"Age: {patient.get('age')}\n"
            f"Gender: {patient.get('gender')}\n"
            f"Condition: {patient.get('condition')}\n"
            f"Symptoms: {', '.join(patient.get('symptoms', []))}\n"
            f"History: {patient.get('history')}\n\n"
            "Perform complete Doogie clinical analysis and return structured insights."
        )

        # ✅ No await (sync client)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": DOOGIE_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )

        ai_output = response.choices[0].message.content
        results.append({
            "id": patient.get("id"),
            "name": patient.get("name"),
            "analysis": ai_output
        })

    return JSONResponse(content={
        "status": "success",
        "total_patients": len(patients),
        "results": results
    })
# ------------------- STEP 4: NICE Care Pathway Bundle Builder -------------------
from care_pathways.bundle_builder import build_bundle  # 👈 Import your builder

class PathwayBuildRequest(BaseModel):
    condition: str

@app.post("/build-pathway-bundle/")
async def build_pathway_bundle(request: PathwayBuildRequest):
    """
    Build a NICE Care Pathway Bundle for a specific condition.
    Example body:
    {
      "condition": "asthma"
    }
    """
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
async def root():
    return {"message": "🚀 Doogie AI PDF API running with Translation, Analysis, and Reference Extraction!"}

# ------------------- Run -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    