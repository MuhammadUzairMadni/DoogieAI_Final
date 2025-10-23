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
# - Deployable on Railway / Serverless (with caveats)
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

# ------------------- Response size config -------------------
# Configure maximum tokens for OpenAI responses. Default 512 (adjustable via env).
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

# Also limit how many characters of long documents we send in the system prompt/context
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3000"))

# ------------------- Load Doogie Master Prompt -------------------
def load_master_prompt(docx_path: str) -> str:
    """Load and combine all paragraphs from the Doogie Master Prompt Word file."""
    try:
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        # fallback (do not crash server); we log minimally and continue
        return "Doogie AI: provide concise, evidence-based medical answers in a clear, professional style."

MASTER_PROMPT_PATH = "Doogie_Master_Prompt.docx"
MASTER_PROMPT_TEXT = load_master_prompt(MASTER_PROMPT_PATH)
print("✅ Doogie Master Prompt loaded (or fallback used).")

# ------------------- Load NHS & Oxford Knowledge -------------------
def load_reference_data():
    """Load all NHS and Oxford guideline data from knowledge/ folder."""
    knowledge_dir = Path("knowledge")
    nhs_data, oxford_data = "", ""
    if knowledge_dir.exists():
        for file in knowledge_dir.glob("*.txt"):
            try:
                contents = file.read_text(encoding="utf-8")
            except Exception:
                contents = ""
            if "nhs" in file.name.lower():
                nhs_data += f"\n\n{file.name}:\n" + contents
            elif "oxford" in file.name.lower():
                oxford_data += f"\n\n{file.name}:\n" + contents
    return nhs_data, oxford_data

NHS_DATA, OXFORD_DATA = load_reference_data()
print("✅ NHS & Oxford reference data loaded (or empty).")

# ------------------- FastAPI Setup -------------------
app = FastAPI(title="Doogie AI - Medical Reasoning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Helper to call OpenAI safely -------------------
def openai_chat(messages, temperature=0.0, max_tokens=None):
    """Wrap OpenAI ChatCompletion call with consistent params and error handling."""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or OPENAI_MAX_TOKENS
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Return a safe message (don't expose raw exception to user)
        return None

# ------------------- Language Detection -------------------
def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    prompt = f"Detect the language of this text and respond only with the language name:\n\n{text[:1000]}"
    result = openai_chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=20)
    if not result:
        return "unknown"
    # normalize common variations (so downstream checks work)
    return result.strip().lower()

# ------------------- Translation Helpers -------------------
def translate_to_english(text: str, source_lang: str) -> str:
    if not text:
        return ""
    try:
        if source_lang and source_lang.lower() in ("english", "en", "unknown"):
            return text
    except Exception:
        pass
    prompt = f"Translate this {source_lang} medical text into clear, concise English, preserving exact clinical meaning:\n\n{text}"
    result = openai_chat(
        [
            {"role": "system", "content": "You are a professional medical translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return result or text  # fallback to original if translation fails

def translate_to_original(text: str, target_lang: str) -> str:
    if not text:
        return ""
    try:
        if target_lang and target_lang.lower() in ("english", "en", "unknown"):
            return text
    except Exception:
        pass
    prompt = f"Translate this English medical text into {target_lang}, preserving accuracy and clinical meaning. Keep it concise:\n\n{text}"
    result = openai_chat(
        [
            {"role": "system", "content": "You are a precise multilingual translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return result or text

# ------------------- Dynamic Knowledge Generator -------------------
def generate_nhs_style_data(topic: str) -> str:
    """Generate NHS/Oxford-style summary for unknown conditions."""
    if not topic:
        return "No topic provided."
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
    Ensure information is medically responsible and evidence-based and keep it concise.
    """
    result = openai_chat([{"role": "system", "content": prompt}], temperature=0.2, max_tokens=OPENAI_MAX_TOKENS)
    return result or "Could not generate dynamic knowledge at this time."

# ------------------- Reasoning Function -------------------
def analyze_with_doogie(prompt_text: str) -> str:
    """Use stored or dynamically generated knowledge + Master Prompt."""
    # Step 1: detect whether to reference stored data
    found_data = ""
    if prompt_text and any(k in prompt_text.lower() for k in ["nhs", "oxford"]):
        found_data = (NHS_DATA + "\n\n" + OXFORD_DATA).strip()

    # Step 2: prepare context (truncate long texts to avoid exceeding model context)
    if not found_data:
        dynamic_data = generate_nhs_style_data(prompt_text)
        context_sources = f"NHS & Oxford Dynamic Knowledge (Generated):\n{dynamic_data}"
    else:
        # truncate to limit characters sent to model
        context_sources = f"NHS References:\n{(NHS_DATA or '')[:MAX_CONTEXT_CHARS]}\n\nOxford Medical Knowledge:\n{(OXFORD_DATA or '')[:MAX_CONTEXT_CHARS]}"

    master = (MASTER_PROMPT_TEXT or "")[:MAX_CONTEXT_CHARS]

    # Build the system context (short & focused)
    system_context = f"{master}\n\n{context_sources}"[:MAX_CONTEXT_CHARS * 2]

    user_query = f"Analyze this clinical query and provide a concise, evidence-aware answer:\n\n{prompt_text}"

    # Step 3: Query model
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_query}
    ]

    result = openai_chat(messages, temperature=0.0, max_tokens=OPENAI_MAX_TOKENS)
    if not result:
        return "Sorry — the reasoning service is temporarily unavailable. Please try again later."

    # Add a short medical-disclaimer / safe note
    disclaimer = ("\n\nNote: This information is for educational purposes and does not replace "
                  "professional medical advice. If this is an emergency, seek immediate care.")

    # Keep final message concise (don't exceed token budget when translated later)
    return result.strip() + disclaimer

# ------------------- Request Model -------------------
class PromptRequest(BaseModel):
    query: str

# ------------------- Main Endpoint -------------------
@app.post("/ask")
def ask_doogie(request: PromptRequest):
    """Accepts patient prompt in any language, reasons with static or dynamic data, replies in same language."""
    user_input = (request.query or "").strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided.")

    detected_lang = detect_language(user_input)  # normalized to lowercase or 'unknown'
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

# ------------------- Run (local dev only) -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
