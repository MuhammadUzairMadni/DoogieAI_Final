# ==============================================================
# Doogie AI - Multi-language Medical Reasoning API (Enhanced)
# ==============================================================
# Features:
# - Accepts user prompts in any language
# - Detects and translates language automatically
# - Uses NHS/Oxford stored knowledge if available
# - Dynamically generates NHS/Oxford-style data if not available
# - Responds in the user's original language
# - Integrated with Doogie Master Prompt
# - Deployable on Railway
# ==============================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from docx import Document
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os

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
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        raise RuntimeError(f"❌ Error loading master prompt: {str(e)}")

MASTER_PROMPT_PATH = "Doogie_Master_Prompt.docx"
MASTER_PROMPT_TEXT = load_master_prompt(MASTER_PROMPT_PATH)
print("✅ Doogie Master Prompt loaded successfully!")

# ------------------- Load NHS & Oxford Knowledge -------------------
def load_reference_data():
    """Load all NHS and Oxford guideline data from knowledge/ folder."""
    knowledge_dir = Path("knowledge")
    nhs_data, oxford_data = "", ""
    for file in knowledge_dir.glob("*.txt"):
        if "nhs" in file.name.lower():
            nhs_data += f"\n\n{file.name}:\n" + file.read_text(encoding="utf-8")
        elif "oxford" in file.name.lower():
            oxford_data += f"\n\n{file.name}:\n" + file.read_text(encoding="utf-8")
    return nhs_data, oxford_data

NHS_DATA, OXFORD_DATA = load_reference_data()
print("✅ NHS & Oxford reference data loaded successfully!")

# ------------------- FastAPI Setup -------------------
app = FastAPI(title="Doogie AI - Medical Reasoning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Language Detection -------------------
def detect_language(text: str) -> str:
    if not text.strip():
        return "Unknown"
    prompt = f"Detect the language of this text and respond only with the language name:\n\n{text[:1000]}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error detecting language: {str(e)}"

# ------------------- Translation Helpers -------------------
def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang.lower() == "english":
        return text
    prompt = f"Translate this {source_lang} medical text into English, preserving meaning:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional medical translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def translate_to_original(text: str, target_lang: str) -> str:
    if target_lang.lower() == "english":
        return text
    prompt = f"Translate this English text into {target_lang}:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a precise multilingual translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ------------------- Dynamic Knowledge Generator -------------------
def generate_nhs_style_data(topic: str) -> str:
    """Generate NHS/Oxford-style summary for unknown conditions."""
    prompt = f"""
    You are a medical assistant trained on NHS and Oxford guidelines.
    Generate a structured summary about '{topic}' using the same style, tone, and structure as NHS and Oxford handbooks.
    Include:
    - Overview
    - Common Symptoms
    - Causes
    - Diagnosis and Tests
    - Treatment and Management
    - When to Seek Emergency Help
    Ensure information is medically responsible and evidence-based.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating dynamic knowledge: {str(e)}"

# ------------------- Reasoning Function -------------------
def analyze_with_doogie(prompt_text: str) -> str:
    """Use stored or dynamically generated knowledge + Master Prompt."""
    # Step 1: Try to detect if known condition is in stored data
    found_data = ""
    for keyword in ["nhs", "oxford"]:
        if keyword in prompt_text.lower():
            found_data = NHS_DATA + "\n\n" + OXFORD_DATA
            break

    # Step 2: If condition not in knowledge base, generate dynamically
    if not found_data:
        dynamic_data = generate_nhs_style_data(prompt_text)
        context_sources = f"""
        NHS & Oxford Dynamic Knowledge (Generated):
        {dynamic_data}
        """
    else:
        context_sources = f"""
        NHS References:
        {NHS_DATA[:2500]}

        Oxford Medical Knowledge:
        {OXFORD_DATA[:2500]}
        """

    # Step 3: Merge with Master Prompt
    context = f"""
    {MASTER_PROMPT_TEXT}

    {context_sources}
    """

    # Step 4: Send to GPT for reasoning
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": f"Analyze this query:\n{prompt_text}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# ------------------- Request Model -------------------
class PromptRequest(BaseModel):
    query: str

# ------------------- Main Endpoint -------------------
@app.post("/ask")
def ask_doogie(request: PromptRequest):
    """Accepts patient prompt in any language, reasons with static or dynamic data, replies in same language."""
    user_input = request.query.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided.")

    detected_lang = detect_language(user_input)
    translated_query = translate_to_english(user_input, detected_lang)
    analysis_result = analyze_with_doogie(translated_query)
    final_response = translate_to_original(analysis_result, detected_lang)

    return JSONResponse(content={
        "language_detected": detected_lang,
        "translated_query": translated_query,
        "response": final_response
    })

# ------------------- Root -------------------
@app.get("/")
def root():
    return {"message": "🚀 Doogie AI (Enhanced) running with Dynamic NHS/Oxford-style reasoning!"}

# ------------------- Run -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
