"""
Brand AI Integrity Tool - v2.0

Misura la Brand Integrity confrontando risposte AI (Gemini, ChatGPT)
con risposte ground truth fornite dall'utente.
Entrambe le AI usano Brave Search per accesso web uniforme.
"""

import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import requests
import json
import time
import os
import smtplib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Dict, List, Optional, Tuple
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

MATCH_THRESHOLD = 0.75

SOCIAL_OPTIONS = ["Instagram", "Facebook", "LinkedIn", "TikTok", "YouTube", "X (Twitter)"]

QUESTIONS = [
    {
        "id": "products",
        "label": "Indica massimo 3 prodotti/servizi principali di {BRAND_NAME}",
        "type": "text",
        "ai_prompt": "Quali sono i 3 principali prodotti o servizi offerti da {BRAND_NAME}? Elenca solo i 3 piu importanti.",
    },
    {
        "id": "sector",
        "label": "In che settore opera {BRAND_NAME}?",
        "type": "text",
        "ai_prompt": "In quale settore opera {BRAND_NAME}?",
        "prefill_from": "sector",
    },
    {
        "id": "target",
        "label": "Qual e il pubblico target principale di {BRAND_NAME}?",
        "type": "text",
        "ai_prompt": "Qual e il pubblico target principale di {BRAND_NAME}?",
    },
    {
        "id": "locations",
        "label": "{BRAND_NAME} ha sedi operative? Se si, dove?",
        "type": "text",
        "ai_prompt": "{BRAND_NAME} ha sedi operative? Se si, dove si trovano?",
    },
    {
        "id": "social",
        "label": "Quali sono i canali social ufficiali di {BRAND_NAME}?",
        "type": "checkbox",
        "options": SOCIAL_OPTIONS,
        "ai_prompt": "Quali sono i canali social ufficiali del brand {BRAND_NAME}? Elenca solo quelli effettivamente attivi.",
    },
    {
        "id": "website",
        "label": "Qual e il sito web ufficiale di {BRAND_NAME}?",
        "type": "text",
        "ai_prompt": "Qual e il sito web ufficiale di {BRAND_NAME}?",
    },
]

LOADING_MESSAGES_GEMINI = [
    "Gemini sta studiando il tuo brand... 🔮",
    "Gemini consulta il web per te... 🌐",
    "Gemini elabora la risposta... ⚡",
    "Gemini ragiona sui dati trovati... 🧠",
]

LOADING_MESSAGES_CHATGPT = [
    "ChatGPT sta cercando informazioni... 💬",
    "ChatGPT naviga il web per te... 🕵️",
    "ChatGPT analizza i risultati... 🔍",
    "ChatGPT prepara la risposta... ✨",
]

LOADING_MESSAGES_EVAL = [
    "Confrontiamo le risposte... 🤔",
    "Il giudice AI valuta la coerenza... ⚖️",
    "Calcoliamo il tuo score... 📊",
    "Quasi fatto, ultimi ritocchi... 🎯",
]


# ============================================================
# CSS & ANIMATIONS
# ============================================================

CUSTOM_CSS = """
<style>
/* === ANIMAZIONI === */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-40px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.03); }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes countUp {
    from { opacity: 0; transform: scale(0.5); }
    to { opacity: 1; transform: scale(1); }
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* === CARD BASE === */
.brand-card {
    background: linear-gradient(135deg, rgba(30,35,50,0.95), rgba(22,27,39,0.98));
    border-radius: 16px;
    padding: 28px;
    margin: 16px 0;
    border: 1px solid rgba(232,119,34,0.15);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    animation: fadeInUp 0.6s ease-out;
}
.brand-card:hover {
    border-color: rgba(232,119,34,0.35);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    transition: all 0.3s ease;
}

/* === SCORE CARD === */
.score-card {
    border-radius: 20px;
    padding: 40px 30px;
    text-align: center;
    animation: fadeInUp 0.8s ease-out;
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%);
    pointer-events: none;
}
.score-card h1 {
    font-size: 4.5em;
    font-weight: 800;
    margin: 10px 0;
    color: white;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    animation: countUp 1s ease-out;
}
.score-card h3 {
    color: rgba(255,255,255,0.9);
    font-size: 1.1em;
    margin: 0;
    font-weight: 400;
}
.score-card h2 {
    color: white;
    font-size: 1.4em;
    margin: 8px 0 0;
    font-weight: 700;
}

/* === AI CARD === */
.ai-card {
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    min-height: 140px;
    animation: fadeInUp 0.6s ease-out;
    position: relative;
    overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.ai-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.4);
}
.ai-card h3 { color: white; font-size: 1.15em; margin: 0; font-weight: 500; }
.ai-card h1 { color: white; font-size: 2.8em; margin: 10px 0 0; font-weight: 800; animation: countUp 1.2s ease-out; }

/* === MESSAGE CARD === */
.message-card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 1.1em;
    animation: fadeIn 0.5s ease-out;
    margin: 12px 0;
}

/* === LOADING === */
.loading-card {
    padding: 24px;
    border-radius: 14px;
    text-align: center;
    margin: 16px 0;
    animation: pulse 2s ease-in-out infinite;
}

/* === INTRO === */
.intro-box {
    background: linear-gradient(135deg, #1a1f2e 0%, #0e1117 100%);
    border: 2px solid rgba(232,119,34,0.3);
    border-radius: 16px;
    padding: 35px;
    margin: 20px 0;
    animation: fadeIn 1s ease-out;
}
.intro-box h2 {
    color: #E87722;
    margin: 0 0 15px;
    font-size: 1.5em;
}
.intro-box p {
    color: #c8c8d8;
    font-size: 1.08em;
    line-height: 1.7;
    margin: 0;
}

/* === DETAIL CARD === */
.detail-card {
    background: rgba(30,35,50,0.7);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    border-left: 4px solid;
    animation: slideInLeft 0.5s ease-out;
}

/* === CTA === */
.cta-box {
    background: linear-gradient(135deg, #E87722 0%, #F57C00 50%, #FF9800 100%);
    background-size: 200% 200%;
    animation: gradientShift 4s ease infinite;
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    margin: 30px 0;
    box-shadow: 0 8px 30px rgba(232,119,34,0.3);
}
.cta-box h2 { color: white; margin: 0 0 15px; font-size: 1.5em; }
.cta-box p { color: rgba(255,255,255,0.92); font-size: 1.1em; margin: 0 0 25px; }
.cta-box a {
    display: inline-block;
    background: white;
    color: #E87722;
    font-weight: 700;
    font-size: 1.1em;
    padding: 14px 36px;
    border-radius: 10px;
    text-decoration: none;
    transition: transform 0.2s, box-shadow 0.2s;
}
.cta-box a:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
}

/* === RECOMMENDATION BOX === */
.reco-box {
    background: linear-gradient(135deg, rgba(30,35,50,0.95), rgba(22,27,39,0.98));
    border: 2px solid rgba(232,119,34,0.4);
    border-radius: 16px;
    padding: 30px;
    margin: 20px 0;
    animation: fadeInUp 0.8s ease-out;
}
.reco-box h3 { color: #E87722; margin: 0 0 20px; font-size: 1.3em; }

/* === BOTTONE PRIMARIO === */
.stButton > button[kind="primary"] {
    background-color: #E87722 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    letter-spacing: 0.02em;
    border-radius: 10px !important;
    padding: 0.6em 2em !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #cf6610 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(232,119,34,0.4) !important;
}

/* === CAMPI INPUT === */
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: #E87722 !important;
    box-shadow: 0 0 0 2px rgba(232,119,34,0.22) !important;
}
.stTextArea label, .stTextInput label {
    font-weight: 700 !important;
    font-size: 0.97em !important;
}
.stTextArea textarea {
    color: #e8e8f0 !important;
    font-size: 0.96em !important;
    line-height: 1.5 !important;
}
.stTextArea textarea:disabled {
    color: #9090a8 !important;
    opacity: 1 !important;
}

/* === PROGRESS BAR === */
.stProgress > div > div {
    background-color: #E87722 !important;
}

/* === EMAIL FORM === */
.email-section {
    background: rgba(30,35,50,0.8);
    border: 1px solid rgba(232,119,34,0.25);
    border-radius: 14px;
    padding: 28px;
    margin: 24px 0;
    animation: fadeInUp 0.7s ease-out;
}
</style>
"""


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    defaults = {
        'current_step': 1,
        'brand_name': '',
        'sector': '',
        'user_answers': {},
        'ai_answers': {},
        'eval_results': {},
        'summary': None,
        'recommendation': None,
        'qualitative_comment': None,
        'api_calls_count': 0,
        'last_api_call_time': 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================
# UTILITIES
# ============================================================

def get_secret(key: str, default: str = None) -> Optional[str]:
    try:
        value = st.secrets[key]
        if value:
            return value
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get(key, default)


def check_secrets() -> Tuple[bool, Optional[str]]:
    required = {
        "GEMINI_API_KEY": "GEMINI_API_KEY",
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "BRAVE_API_KEY": "BRAVE_API_KEY",
    }
    for key, label in required.items():
        val = get_secret(key)
        if not val or val.startswith("YOUR_"):
            return False, f"{label} non configurata"
    return True, None


def rate_limit_check():
    now = time.time()
    if now - st.session_state.last_api_call_time > 60:
        st.session_state.api_calls_count = 0
        st.session_state.last_api_call_time = now
    if st.session_state.api_calls_count >= 30:
        remaining = int(60 - (now - st.session_state.last_api_call_time))
        return False, f"Rate limit raggiunto. Riprova tra {remaining}s"
    st.session_state.api_calls_count += 1
    return True, None


def get_color_for_score(score: int) -> str:
    if score >= 80:
        return "#4CAF50"
    elif score >= 60:
        return "#FF9800"
    return "#F44336"


def get_judgment(score: int) -> str:
    if score >= 80:
        return "ECCELLENTE"
    elif score >= 60:
        return "BUONO"
    return "SCARSO"


# ============================================================
# AI MODEL CONFIGURATION
# ============================================================

def configure_models():
    try:
        genai.configure(api_key=get_secret("GEMINI_API_KEY"))

        gemini_name = get_secret("GEMINI_MODEL", "gemini-2.0-flash")
        evaluator_name = get_secret("EVALUATOR_MODEL", gemini_name)

        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        gemini_model = genai.GenerativeModel(
            model_name=gemini_name,
            generation_config={"temperature": 0.2, "top_p": 0.8, "top_k": 40, "max_output_tokens": 2048},
            safety_settings=safety,
        )

        evaluator_model = genai.GenerativeModel(
            model_name=evaluator_name,
            generation_config={"temperature": 0.2, "top_p": 0.8, "top_k": 40, "max_output_tokens": 2048, "response_mime_type": "application/json"},
            safety_settings=safety,
        )

        openai_client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

        return gemini_model, openai_client, evaluator_model, None
    except Exception as e:
        return None, None, None, f"Errore configurazione AI: {str(e)}"


# ============================================================
# WEB SEARCH (BRAVE) -- usato da entrambe le AI
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def brave_search(query: str, max_results: int = 10) -> Tuple[str, bool]:
    try:
        brave_key = get_secret("BRAVE_API_KEY", "")
        if not brave_key:
            return "", False

        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": brave_key},
            params={"q": query, "count": max_results, "search_lang": "it", "text_decorations": False, "safesearch": "moderate"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        if not results:
            return "", False

        formatted = ""
        for i, r in enumerate(results[:max_results], 1):
            formatted += f"{i}. {r.get('title', '')}\n{r.get('description', '')}\nFonte: {r.get('url', '')}\n\n"
        return formatted.strip(), True
    except Exception:
        return "", False


# ============================================================
# AI ANSWER GENERATION (entrambe con Brave Search)
# ============================================================

def _build_prompt_with_search(brand_name: str, question: str, search_results: str, search_ok: bool) -> str:
    if search_ok and search_results:
        return f"""Rispondi alla seguente domanda su {brand_name}.

Ho effettuato una ricerca web e ho trovato queste informazioni:

{search_results}

Domanda: {question}

ISTRUZIONI:
- Usa PRINCIPALMENTE le informazioni trovate sul web
- Integra con la tua conoscenza solo se necessario
- Se le informazioni web sono insufficienti, specifica che stai usando la tua conoscenza generale
- Rispondi in italiano, in modo chiaro e diretto (massimo 200 parole)
- Non menzionare la ricerca web nella risposta, rispondi in modo naturale"""
    else:
        return f"""Rispondi alla seguente domanda su {brand_name}.

Domanda: {question}

ISTRUZIONI:
- Usa la tua conoscenza per rispondere
- Se non hai informazioni certe, specificalo
- Rispondi in italiano, in modo chiaro e diretto (massimo 200 parole)"""


@st.cache_data(ttl=300, show_spinner=False)
def generate_gemini_answer(brand_name: str, question: str, _model_name: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        genai.configure(api_key=get_secret("GEMINI_API_KEY"))
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        model = genai.GenerativeModel(
            model_name=_model_name,
            generation_config={"temperature": 0.2, "top_p": 0.8, "top_k": 40, "max_output_tokens": 2048},
            safety_settings=safety,
        )

        search_query = f"{brand_name} {question.replace('{BRAND_NAME}', brand_name)}"
        search_results, search_ok = brave_search(search_query)

        final_q = question.replace("{BRAND_NAME}", brand_name)
        prompt = _build_prompt_with_search(brand_name, final_q, search_results, search_ok)

        response = model.generate_content(prompt, request_options={"timeout": 30})
        if not response.candidates:
            return None, "Risposta bloccata dai safety filters"
        if response.text:
            return response.text.strip(), None
        return None, "Risposta vuota da Gemini"
    except Exception as e:
        return None, f"Errore Gemini: {str(e)}"


@st.cache_data(ttl=300, show_spinner=False)
def generate_openai_answer(brand_name: str, question: str, _model_name: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

        search_query = f"{brand_name} {question.replace('{BRAND_NAME}', brand_name)}"
        search_results, search_ok = brave_search(search_query)

        final_q = question.replace("{BRAND_NAME}", brand_name)
        prompt = _build_prompt_with_search(brand_name, final_q, search_results, search_ok)

        openai_model = get_secret("OPENAI_MODEL", "gpt-4o-mini")

        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "Rispondi in modo naturale e diretto in italiano."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        if response and response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip(), None
        return None, "Risposta vuota da ChatGPT"
    except Exception as e:
        return None, f"Errore ChatGPT: {str(e)}"


# ============================================================
# RECOMMENDATION (domanda nascosta)
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def generate_recommendation(sector: str, ai_name: str, _model_identifier: str) -> Tuple[Optional[str], Optional[str]]:
    """Chiede all'AI: 'Nel settore X, quale brand consiglieresti?' SENZA menzionare il brand."""
    prompt = f"""Nel settore "{sector}" in Italia, quale brand consiglieresti e perche?

Rispondi in modo diretto e naturale in italiano (massimo 150 parole).
Indica un brand specifico con una breve motivazione."""

    try:
        if ai_name == "gemini":
            genai.configure(api_key=get_secret("GEMINI_API_KEY"))
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            safety = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
            model = genai.GenerativeModel(
                model_name=_model_identifier,
                generation_config={"temperature": 0.3, "max_output_tokens": 1024},
                safety_settings=safety,
            )
            resp = model.generate_content(prompt, request_options={"timeout": 30})
            if resp and resp.text:
                return resp.text.strip(), None
            return None, "Risposta vuota"
        elif ai_name == "openai":
            client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=_model_identifier,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            if resp and resp.choices:
                return resp.choices[0].message.content.strip(), None
            return None, "Risposta vuota"
    except Exception as e:
        return None, str(e)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_batch(evaluator, question: str, ai_answers: Dict[str, str], user_answer: str, retry: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
    ai_names = list(ai_answers.keys())
    if not ai_names:
        return {}, None

    answers_block = ""
    for name in ai_names:
        label = {"gemini": "Gemini", "openai": "ChatGPT"}.get(name, name)
        answers_block += f"\nRisposta {label}:\n{ai_answers[name]}\n"

    prompt = f"""Valuta la coerenza tra le risposte AI e la risposta ground truth (utente).

Domanda: {question}
{answers_block}
Risposta ground truth (utente):
{user_answer}

Criteri di valutazione:
- "corretta" (score >= 0.75) se semanticamente allineata alla ground truth e non contraddice
- "parziale" (score 0.5-0.74) se le info principali sono corrette MA aggiunge molti dettagli non presenti nella ground truth
- "sbagliata" (score < 0.5) se contraddice o manca elementi essenziali

IMPORTANTE: Se la risposta AI aggiunge molti dettagli specifici non verificabili oltre la ground truth, penalizza di 0.10-0.25.

{"GENERA SOLO JSON VALIDO. Ogni 'reason' max 100 caratteri." if retry else ""}

Restituisci JSON:
{{
  {', '.join(f'"{n}": {{"score": 0.85, "is_correct": true, "reason": "Breve spiegazione", "key_conflicts": []}}' for n in ai_names)}
}}

Schema per ogni AI:
- score: 0.0-1.0
- is_correct: true se score >= {MATCH_THRESHOLD}
- reason: stringa breve (max 100 char)
- key_conflicts: array stringhe (max 3, puo essere [])"""

    try:
        response = evaluator.generate_content(prompt, request_options={"timeout": 30})
        if not response or not response.candidates:
            return None, "Risposta bloccata o vuota dall'evaluator"

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return None, "Risposta vuota dall'evaluator"

        text = candidate.content.parts[0].text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]).replace("```json", "").replace("```", "").strip()

        batch = json.loads(text)
        results = {}
        for name in ai_names:
            if name in batch:
                r = batch[name]
                r["score"] = float(r.get("score", 0))
                r["is_correct"] = bool(r.get("is_correct", False))
                if isinstance(r.get("reason"), str):
                    r["reason"] = r["reason"].replace("\n", " ").strip()
                r.setdefault("key_conflicts", [])
                results[name] = r
        return results, None
    except json.JSONDecodeError:
        if not retry:
            return evaluate_batch(evaluator, question, ai_answers, user_answer, retry=True)
        return None, "Errore parsing JSON dall'evaluator"
    except Exception as e:
        return None, f"Errore valutazione: {str(e)}"


def generate_qualitative_comment(evaluator, brand_name: str, sector: str, summary: Dict, eval_results: Dict, questions: list) -> str:
    """Genera un commento qualitativo con suggerimenti."""
    wrong_questions = []
    for idx, res in eval_results.items():
        if not res.get("is_correct", False):
            q_label = questions[idx]["label"].replace("{BRAND_NAME}", brand_name) if idx < len(questions) else f"Domanda {idx+1}"
            wrong_questions.append(q_label)

    prompt = f"""Sei un esperto di brand reputation e AI.
Analizza questi risultati del Brand AI Integrity Score per "{brand_name}" (settore: {sector}):

- Score complessivo: {summary['integrity_score']}/100
- Score Gemini: {summary['ai_scores'].get('gemini', 0)}/100
- Score ChatGPT: {summary['ai_scores'].get('openai', 0)}/100
- Domande dove le AI hanno sbagliato: {', '.join(wrong_questions) if wrong_questions else 'Nessuna'}

Genera un commento qualitativo in italiano (4-5 frasi) che:
1. Commenta il risultato generale
2. Evidenzia le aree critiche
3. Suggerisce 2-3 azioni concrete per migliorare la rappresentazione del brand nelle AI

Scrivi in modo professionale ma accessibile. Non usare elenchi puntati, scrivi in paragrafi fluidi."""

    try:
        response = evaluator.generate_content(prompt, request_options={"timeout": 30})
        if response and response.text:
            return response.text.strip()
    except Exception:
        pass
    return "Analisi qualitativa non disponibile al momento."


# ============================================================
# PDF GENERATION
# ============================================================

def generate_pdf_report(brand_name, sector, summary, eval_results, questions_list, user_answers, ai_answers, recommendation, qualitative_comment):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=30)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=26, textColor=colors.HexColor('#E87722'),
                                  spaceAfter=10, alignment=TA_CENTER, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=13, textColor=colors.HexColor('#666'),
                                     spaceAfter=20, alignment=TA_CENTER)
    heading_style = ParagraphStyle('H', parent=styles['Heading2'], fontSize=17, textColor=colors.HexColor('#E87722'),
                                    spaceAfter=14, fontName='Helvetica-Bold')
    subheading_style = ParagraphStyle('SH', parent=styles['Heading3'], fontSize=13, textColor=colors.HexColor('#333'),
                                       spaceAfter=10, fontName='Helvetica-Bold')
    box_style = ParagraphStyle('BX', parent=styles['Normal'], fontSize=9, leading=13)
    normal = styles['Normal']

    story = []

    # Title
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("BRAND AI INTEGRITY REPORT", title_style))
    story.append(Paragraph(f"Brand: {brand_name} | Settore: {sector}", subtitle_style))
    story.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y - %H:%M')}", subtitle_style))
    story.append(Spacer(1, 0.4 * inch))

    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    score = summary['integrity_score']
    ai_scores = summary.get('ai_scores', {})

    data = [
        ['METRICA', 'VALORE', 'VALUTAZIONE'],
        ['Brand AI Integrity Score', f"{score}/100", get_judgment(score)],
        ['', '', ''],
        ['Score Gemini', f"{ai_scores.get('gemini', 0)}/100", get_judgment(ai_scores.get('gemini', 0))],
        ['Score ChatGPT', f"{ai_scores.get('openai', 0)}/100", get_judgment(ai_scores.get('openai', 0))],
        ['', '', ''],
        ['Domande Totali', str(summary['total']), ''],
        ['Corrette (media)', str(summary['correct']), ''],
        ['Da Migliorare (media)', str(summary['incorrect']), ''],
    ]

    t = Table(data, colWidths=[2.5 * inch, 1.5 * inch, 2 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E87722')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor(get_color_for_score(score))),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.white),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 13),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor(get_color_for_score(ai_scores.get('gemini', 0)))),
        ('TEXTCOLOR', (0, 3), (-1, 3), colors.white),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor(get_color_for_score(ai_scores.get('openai', 0)))),
        ('TEXTCOLOR', (0, 4), (-1, 4), colors.white),
        ('BACKGROUND', (0, 6), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    # Qualitative comment
    if qualitative_comment:
        story.append(Paragraph("ANALISI QUALITATIVA", heading_style))
        story.append(Paragraph(qualitative_comment, normal))
        story.append(Spacer(1, 0.3 * inch))

    # Details
    story.append(PageBreak())
    story.append(Paragraph("ANALISI DETTAGLIATA", heading_style))

    for idx in sorted(eval_results.keys()):
        if idx >= len(questions_list):
            continue
        result = eval_results[idx]
        q = questions_list[idx]
        question_text = q["label"].replace("{BRAND_NAME}", brand_name)
        avg_score = result.get('average_score', 0)
        is_correct = result.get('is_correct', False)

        story.append(Paragraph(f"<b>DOMANDA {idx + 1}:</b> {question_text}", subheading_style))

        status = "CORRETTA" if is_correct else "DA MIGLIORARE"
        sc = '#4CAF50' if is_correct else '#F44336'
        st_data = [['Score', f"{avg_score:.2f}/1.00", status]]
        st_table = Table(st_data, colWidths=[1.5 * inch, 1.5 * inch, 2 * inch])
        st_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(sc)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ]))
        story.append(st_table)
        story.append(Spacer(1, 0.1 * inch))

        # Ground truth
        story.append(Paragraph("<b>Ground Truth:</b>", normal))
        gt_para = Paragraph(str(user_answers.get(idx, "N/A")), box_style)
        gt_table = Table([[gt_para]], colWidths=[4.5 * inch])
        gt_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E8F5E9')),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#4CAF50')),
            ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10), ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(gt_table)
        story.append(Spacer(1, 0.15 * inch))

        # AI answers
        if idx in ai_answers:
            for ai_name, ai_label in [("gemini", "Gemini"), ("openai", "ChatGPT")]:
                if ai_name in ai_answers[idx] and ai_name in result:
                    ai_res = result[ai_name]
                    ai_sc = ai_res.get('score', 0)
                    bg = '#E8F5E9' if ai_sc >= 0.75 else ('#FFF3E0' if ai_sc >= 0.5 else '#FFEBEE')
                    bd = '#4CAF50' if ai_sc >= 0.75 else ('#FF9800' if ai_sc >= 0.5 else '#F44336')

                    story.append(Paragraph(f"<b>{ai_label}</b> (Score: {ai_sc:.2f})", normal))
                    ai_para = Paragraph(ai_answers[idx][ai_name], box_style)
                    ai_table = Table([[ai_para]], colWidths=[4.5 * inch])
                    ai_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor(bd)),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                        ('TOPPADDING', (0, 0), (-1, -1), 10), ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ]))
                    story.append(ai_table)
                    story.append(Paragraph(f"<i>{ai_res.get('reason', '')}</i>", normal))
                    story.append(Spacer(1, 0.1 * inch))

        story.append(Spacer(1, 0.3 * inch))

    # Recommendation
    if recommendation:
        story.append(PageBreak())
        story.append(Paragraph("DOMANDA CHIAVE: CHI CONSIGLIANO LE AI?", heading_style))
        story.append(Paragraph(f"Domanda: \"Nel settore {sector}, quale brand consiglieresti?\"", normal))
        story.append(Spacer(1, 0.15 * inch))
        for ai_name, ai_label in [("gemini", "Gemini"), ("openai", "ChatGPT")]:
            if ai_name in recommendation:
                story.append(Paragraph(f"<b>{ai_label}:</b>", normal))
                story.append(Paragraph(recommendation[ai_name], box_style))
                story.append(Spacer(1, 0.15 * inch))

    # Footer + CTA
    story.append(PageBreak())
    story.append(Spacer(1, 1 * inch))
    story.append(Paragraph("Report generato da Brand AI Integrity", normal))
    story.append(Paragraph("Sviluppato dal <b>Team Innovation di AvantGrade.com</b>", normal))
    story.append(Spacer(1, 0.5 * inch))

    cta_style = ParagraphStyle('CTA', parent=normal, fontSize=12, textColor=colors.white, alignment=TA_CENTER, fontName='Helvetica-Bold')
    cta_text = '<link href="https://www.avantgrade.com/schedule-a-call" color="white">Vuoi migliorare il tuo Brand AI Integrity? Parliamone insieme</link>'
    cta_table = Table([[Paragraph(cta_text, cta_style)]], colWidths=[5 * inch])
    cta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E87722')),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#CF6610')),
        ('TOPPADDING', (0, 0), (-1, -1), 15), ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('LEFTPADDING', (0, 0), (-1, -1), 20), ('RIGHTPADDING', (0, 0), (-1, -1), 20),
    ]))
    story.append(cta_table)

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# EMAIL (SMTP2GO)
# ============================================================

def send_email_report(to_email: str, brand_name: str, pdf_buffer: BytesIO) -> Tuple[bool, str]:
    host = get_secret("SMTP2GO_HOST", "mail.smtp2go.com")
    port = int(get_secret("SMTP2GO_PORT", "587"))
    username = get_secret("SMTP2GO_USERNAME", "")
    password = get_secret("SMTP2GO_PASSWORD", "")
    sender = get_secret("SMTP2GO_SENDER", "noreply@avantgrade.com")

    if not username or not password:
        return False, "Credenziali SMTP2GO non configurate"

    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = to_email
        msg['Subject'] = f"Brand AI Integrity Report - {brand_name}"

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; color: #333; max-width: 600px; margin: 0 auto;">
    <div style="background: linear-gradient(135deg, #E87722, #FF9800); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
        <h1 style="color: white; margin: 0;">Brand AI Integrity Report</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0;">Brand: {brand_name}</p>
    </div>
    <div style="padding: 30px; background: #f9f9f9;">
        <p>Ciao,</p>
        <p>In allegato trovi il report completo del <b>Brand AI Integrity Score</b> per <b>{brand_name}</b>.</p>
        <p>Il report include l'analisi dettagliata delle risposte di Gemini e ChatGPT confrontate con le informazioni corrette del tuo brand.</p>
        <hr style="border: 1px solid #eee; margin: 20px 0;">
        <p style="text-align: center;">
            <a href="https://www.avantgrade.com/schedule-a-call"
               style="background: #E87722; color: white; padding: 12px 30px; text-decoration: none; border-radius: 8px; font-weight: bold;">
               Vuoi migliorare il tuo score? Parliamone
            </a>
        </p>
    </div>
    <div style="padding: 15px; text-align: center; color: #999; font-size: 12px;">
        <p>Report generato da Brand AI Integrity - Team Innovation AvantGrade.com</p>
    </div>
</body>
</html>"""

        msg.attach(MIMEText(html_body, 'html'))

        pdf_buffer.seek(0)
        pdf_attachment = MIMEApplication(pdf_buffer.read(), _subtype='pdf')
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename=f'Brand_AI_Integrity_{brand_name}.pdf')
        msg.attach(pdf_attachment)

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(sender, to_email, msg.as_string())

        return True, "Email inviata con successo!"
    except Exception as e:
        return False, f"Errore invio email: {str(e)}"


# ============================================================
# UI: STEP 1 - BRAND & SECTOR
# ============================================================

def render_step_1():
    # Intro
    st.markdown("""
    <div class="intro-box">
        <h2>Come funziona?</h2>
        <p>
            Questo strumento misura quanto le intelligenze artificiali conoscono davvero il tuo brand.<br><br>
            <b>In 3 semplici passi:</b> inserisci il nome del tuo brand, rispondi a poche domande con le informazioni corrette,
            e scopri se Gemini e ChatGPT raccontano la verita su di te.<br><br>
            Il risultato e il tuo <b>Brand AI Integrity Score</b>: un punteggio da 0 a 100 che ti dice
            quanto sei rappresentato correttamente nel mondo delle AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Iniziamo! Inserisci i dati del tuo brand")

    def _on_brand_change():
        new_val = st.session_state.get("brand_input", "")
        if new_val != st.session_state.brand_name:
            st.session_state.brand_name = new_val
            st.session_state.ai_answers = {}
            st.session_state.user_answers = {}
            st.session_state.eval_results = {}
            st.session_state.summary = None
            st.session_state.recommendation = None
            st.session_state.qualitative_comment = None

    st.text_input(
        "Nome del Brand",
        value=st.session_state.brand_name,
        placeholder="es. Nike, Apple, AvantGrade...",
        key="brand_input",
        on_change=_on_brand_change,
    )

    def _on_sector_change():
        st.session_state.sector = st.session_state.get("sector_input", "")

    st.text_input(
        "Settore di riferimento",
        value=st.session_state.sector,
        placeholder="es. Digital Marketing, Moda, Food & Beverage...",
        key="sector_input",
        on_change=_on_sector_change,
    )

    brand_name = st.session_state.brand_name
    sector = st.session_state.sector

    if brand_name and sector:
        st.markdown(f"""
        <div class="message-card" style="background: rgba(76,175,80,0.15); border: 1px solid rgba(76,175,80,0.4);">
            <span style="color: #4CAF50; font-weight: 700;">✓ Brand: {brand_name}</span> |
            <span style="color: #4CAF50; font-weight: 700;">Settore: {sector}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Continua →", type="primary"):
            # Pre-fill sector answer
            for i, q in enumerate(QUESTIONS):
                if q.get("prefill_from") == "sector":
                    st.session_state.user_answers[i] = sector
            st.session_state.current_step = 2
            st.rerun()
    else:
        st.info("Compila entrambi i campi per continuare")


# ============================================================
# UI: STEP 2 - QUESTIONS & PROCESSING
# ============================================================

def render_step_2(gemini_model, openai_client, evaluator_model):
    brand_name = st.session_state.brand_name
    sector = st.session_state.sector

    # Back button
    if st.button("← Torna indietro"):
        st.session_state.current_step = 1
        st.rerun()

    st.markdown(f"### Rispondi alle domande su **{brand_name}**")
    st.markdown("Inserisci le risposte corrette secondo il tuo brand. Le AI cercheranno queste informazioni sul web per confrontarle.")

    all_valid = True

    for idx, q in enumerate(QUESTIONS):
        label = q["label"].replace("{BRAND_NAME}", brand_name)

        st.markdown(f"**{label}**")

        if q["type"] == "checkbox":
            # Social channels with checkboxes
            current_selection = st.session_state.user_answers.get(idx, [])
            if isinstance(current_selection, str):
                current_selection = [s.strip() for s in current_selection.split(",") if s.strip()]

            selected = st.multiselect(
                "Seleziona i canali attivi",
                options=q["options"],
                default=current_selection if current_selection else [],
                key=f"social_{idx}",
            )
            st.session_state.user_answers[idx] = ", ".join(selected) if selected else ""
            if not selected:
                all_valid = False

        elif q.get("prefill_from") == "sector":
            # Sector pre-filled and disabled
            st.text_input(
                "Risposta (compilata automaticamente dal settore)",
                value=sector,
                disabled=True,
                key=f"answer_{idx}",
            )
            st.session_state.user_answers[idx] = sector

        else:
            answer = st.text_area(
                "La tua risposta",
                value=st.session_state.user_answers.get(idx, ""),
                height=100,
                placeholder="Scrivi qui la risposta corretta per il tuo brand...",
                key=f"answer_{idx}",
            )
            st.session_state.user_answers[idx] = answer
            if not answer.strip():
                all_valid = False

        st.markdown("---")

    if all_valid:
        st.markdown("""
        <div class="message-card" style="background: rgba(76,175,80,0.15); border: 1px solid rgba(76,175,80,0.4);">
            <span style="color: #4CAF50; font-weight: 700;">✓ Tutte le risposte sono complete!</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Analizza con le AI e Calcola Brand Integrity", type="primary"):
            _run_analysis(gemini_model, openai_client, evaluator_model)
    else:
        st.info("Completa tutte le risposte per procedere con l'analisi")


def _run_analysis(gemini_model, openai_client, evaluator_model):
    brand_name = st.session_state.brand_name
    sector = st.session_state.sector
    questions = QUESTIONS

    gemini_model_name = get_secret("GEMINI_MODEL", "gemini-2.0-flash")
    openai_model_name = get_secret("OPENAI_MODEL", "gpt-4o-mini")

    total_steps = len(questions) * 2 + len(questions) + 2 + 1  # AI gen + eval + reco + comment
    current_step = 0
    start_time = time.time()
    estimated_seconds = len(questions) * 15

    progress_bar = st.progress(0)
    status_container = st.empty()
    errors = []

    st.session_state.ai_answers = {}

    # PHASE 1: Generate AI answers
    for idx, q in enumerate(questions):
        st.session_state.ai_answers[idx] = {}
        ai_prompt = q["ai_prompt"].replace("{BRAND_NAME}", brand_name)

        elapsed = int(time.time() - start_time)
        remaining = max(0, estimated_seconds - elapsed)

        # Gemini
        msg_idx = idx % len(LOADING_MESSAGES_GEMINI)
        status_container.markdown(f"""
        <div class="loading-card" style="background: linear-gradient(135deg, #1565C0, #0D47A1);">
            <h3 style="color: white; margin: 0;">⚫ Gemini -- Domanda {idx + 1}/{len(questions)}</h3>
            <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 1.05em;">{LOADING_MESSAGES_GEMINI[msg_idx]}</p>
            <p style="color: rgba(255,255,255,0.6); margin: 4px 0 0; font-size: 0.9em;">~{remaining}s rimanenti</p>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(current_step / total_steps)

        answer, err = generate_gemini_answer(brand_name, ai_prompt, gemini_model_name)
        if err:
            errors.append(f"Gemini Q{idx + 1}: {err}")
        else:
            st.session_state.ai_answers[idx]["gemini"] = answer
        current_step += 1

        # ChatGPT
        elapsed = int(time.time() - start_time)
        remaining = max(0, estimated_seconds - elapsed)
        msg_idx = idx % len(LOADING_MESSAGES_CHATGPT)
        status_container.markdown(f"""
        <div class="loading-card" style="background: linear-gradient(135deg, #2E7D32, #1B5E20);">
            <h3 style="color: white; margin: 0;">🟢 ChatGPT -- Domanda {idx + 1}/{len(questions)}</h3>
            <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 1.05em;">{LOADING_MESSAGES_CHATGPT[msg_idx]}</p>
            <p style="color: rgba(255,255,255,0.6); margin: 4px 0 0; font-size: 0.9em;">~{remaining}s rimanenti</p>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(current_step / total_steps)

        answer, err = generate_openai_answer(brand_name, ai_prompt, openai_model_name)
        if err:
            errors.append(f"ChatGPT Q{idx + 1}: {err}")
        else:
            st.session_state.ai_answers[idx]["openai"] = answer
        current_step += 1

    # PHASE 2: Evaluate
    st.session_state.eval_results = {}
    ai_models = ["gemini", "openai"]

    for idx in sorted(st.session_state.ai_answers.keys()):
        elapsed = int(time.time() - start_time)
        remaining = max(0, estimated_seconds - elapsed)
        msg_idx = idx % len(LOADING_MESSAGES_EVAL)
        status_container.markdown(f"""
        <div class="loading-card" style="background: linear-gradient(135deg, #BF360C, #E65100);">
            <h3 style="color: white; margin: 0;">📊 Valutazione -- Domanda {idx + 1}/{len(questions)}</h3>
            <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 1.05em;">{LOADING_MESSAGES_EVAL[msg_idx]}</p>
            <p style="color: rgba(255,255,255,0.6); margin: 4px 0 0; font-size: 0.9em;">~{remaining}s rimanenti</p>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(current_step / total_steps)

        q = questions[idx]
        question_text = q["ai_prompt"].replace("{BRAND_NAME}", brand_name)
        ai_ans = st.session_state.ai_answers[idx]
        user_ans = st.session_state.user_answers.get(idx, "")

        batch_results, batch_err = evaluate_batch(evaluator_model, question_text, ai_ans, user_ans)

        st.session_state.eval_results[idx] = {}
        scores = []

        if batch_err:
            errors.append(f"Eval Q{idx + 1}: {batch_err}")
        else:
            for ai_name in ai_models:
                if ai_name in batch_results:
                    st.session_state.eval_results[idx][ai_name] = batch_results[ai_name]
                    scores.append(batch_results[ai_name]['score'])

        if scores:
            avg = sum(scores) / len(scores)
            st.session_state.eval_results[idx]['average_score'] = avg
            st.session_state.eval_results[idx]['is_correct'] = avg >= MATCH_THRESHOLD

        current_step += 1

    # PHASE 3: Recommendation
    status_container.markdown("""
    <div class="loading-card" style="background: linear-gradient(135deg, #6A1B9A, #4A148C);">
        <h3 style="color: white; margin: 0;">🏆 Domanda chiave: chi consigliano le AI?</h3>
        <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0;">Chiediamo alle AI chi consiglierebbero nel tuo settore...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(current_step / total_steps)

    reco = {}
    for ai_name, model_id in [("gemini", gemini_model_name), ("openai", openai_model_name)]:
        answer, err = generate_recommendation(sector, ai_name, model_id)
        if not err and answer:
            reco[ai_name] = answer
        current_step += 1

    st.session_state.recommendation = reco

    # PHASE 4: Calculate summary
    if st.session_state.eval_results:
        total = len(st.session_state.eval_results)
        ai_scores_lists = {ai: [] for ai in ai_models}
        for result in st.session_state.eval_results.values():
            for ai_name in ai_models:
                if ai_name in result and 'score' in result[ai_name]:
                    ai_scores_lists[ai_name].append(result[ai_name]['score'])

        ai_averages = {
            ai: round(sum(sc) / len(sc) * 100) if sc else 0
            for ai, sc in ai_scores_lists.items()
        }

        integrity_score = round(sum(ai_averages.values()) / len(ai_averages)) if ai_averages else 0
        correct = sum(1 for r in st.session_state.eval_results.values() if r.get('is_correct', False))

        st.session_state.summary = {
            'total': total,
            'correct': correct,
            'incorrect': total - correct,
            'integrity_score': integrity_score,
            'ai_scores': ai_averages,
        }

    # PHASE 5: Qualitative comment
    status_container.markdown("""
    <div class="loading-card" style="background: linear-gradient(135deg, #E87722, #CF6610);">
        <h3 style="color: white; margin: 0;">✨ Generazione analisi qualitativa...</h3>
        <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0;">L'AI sta preparando i suggerimenti per te</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.summary:
        comment = generate_qualitative_comment(
            evaluator_model, brand_name, sector,
            st.session_state.summary, st.session_state.eval_results, questions
        )
        st.session_state.qualitative_comment = comment

    progress_bar.progress(1.0)

    total_time = int(time.time() - start_time)
    status_container.markdown(f"""
    <div class="loading-card" style="background: linear-gradient(135deg, #2E7D32, #1B5E20);">
        <h2 style="color: white; margin: 0;">✅ Analisi completata!</h2>
        <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 1.1em;">Tempo totale: {total_time} secondi</p>
    </div>
    """, unsafe_allow_html=True)

    if errors:
        with st.expander(f"⚠️ {len(errors)} avvisi durante l'elaborazione"):
            for err in errors[:10]:
                st.text(err)

    time.sleep(2)

    if st.session_state.eval_results:
        st.session_state.current_step = 3
        st.rerun()
    else:
        st.error("Nessun risultato ottenuto. Controlla gli errori e riprova.")


# ============================================================
# UI: STEP 3 - RESULTS DASHBOARD
# ============================================================

def render_step_3():
    if not st.session_state.summary:
        st.error("Nessun risultato disponibile")
        if st.button("← Torna indietro"):
            st.session_state.current_step = 2
            st.rerun()
        return

    brand_name = st.session_state.brand_name
    sector = st.session_state.sector
    summary = st.session_state.summary
    score = summary['integrity_score']
    ai_scores = summary.get('ai_scores', {})

    # Models info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**⚫ Gemini** | {get_secret('GEMINI_MODEL', 'gemini-2.0-flash')}")
    with col2:
        st.markdown(f"**🟢 ChatGPT** | {get_secret('OPENAI_MODEL', 'gpt-4o-mini')}")

    st.markdown("---")

    # Main score
    color = get_color_for_score(score)
    judgment = get_judgment(score)

    if score >= 80:
        emoji = "🟢"
        message = "Ottimo lavoro: le AI rappresentano il tuo brand in modo chiaro e affidabile! 😎"
    elif score >= 60:
        emoji = "🟡"
        message = "Buono, ma puoi fare di meglio! Alcune imprecisioni sono migliorabili."
    else:
        emoji = "🔴"
        message = "Le AI non rappresentano correttamente il tuo brand. Che ne dici di fare due chiacchiere? 😭"

    # Animated score with JS counter
    score_id = f"score_{uuid.uuid4().hex[:8]}"
    st.markdown(f"""
    <div class="score-card" style="background: linear-gradient(135deg, {color}, {color}dd);">
        <h3>Punteggio complessivo di Brand AI Integrity</h3>
        <h1>{emoji} <span id="{score_id}">0</span>/100</h1>
        <h2>{judgment}</h2>
    </div>
    <script>
    (function() {{
        var target = {score};
        var el = document.getElementById('{score_id}');
        if (!el) return;
        var current = 0;
        var step = target / 60;
        var interval = setInterval(function() {{
            current += step;
            if (current >= target) {{
                el.textContent = target;
                clearInterval(interval);
            }} else {{
                el.textContent = Math.floor(current);
            }}
        }}, 16);
    }})();
    </script>
    """, unsafe_allow_html=True)

    # Message
    st.markdown(f"""
    <div class="message-card" style="border: 2px solid {color}; color: {color}; font-weight: 600;">
        {message}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # AI scores
    st.markdown("### 📈 Performance per AI")
    col1, col2 = st.columns(2)

    gemini_score = ai_scores.get('gemini', 0)
    openai_score = ai_scores.get('openai', 0)
    g_color = get_color_for_score(gemini_score)
    o_color = get_color_for_score(openai_score)

    g_id = f"gs_{uuid.uuid4().hex[:8]}"
    o_id = f"os_{uuid.uuid4().hex[:8]}"

    with col1:
        st.markdown(f"""
        <div class="ai-card" style="background: linear-gradient(135deg, {g_color}, {g_color}cc);">
            <h3>⚫ Gemini</h3>
            <h1><span id="{g_id}">0</span>/100</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="ai-card" style="background: linear-gradient(135deg, {o_color}, {o_color}cc);">
            <h3>🟢 ChatGPT</h3>
            <h1><span id="{o_id}">0</span>/100</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <script>
    (function() {{
        function animateScore(id, target) {{
            var el = document.getElementById(id);
            if (!el) return;
            var current = 0;
            var step = target / 50;
            var interval = setInterval(function() {{
                current += step;
                if (current >= target) {{ el.textContent = target; clearInterval(interval); }}
                else {{ el.textContent = Math.floor(current); }}
            }}, 20);
        }}
        setTimeout(function() {{ animateScore('{g_id}', {gemini_score}); }}, 400);
        setTimeout(function() {{ animateScore('{o_id}', {openai_score}); }}, 600);
    }})();
    </script>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Domande totali", summary['total'])
    with col2:
        st.metric("✅ Corrette (media)", summary['correct'])
    with col3:
        st.metric("❌ Da migliorare (media)", summary['incorrect'])

    st.markdown("---")

    # Qualitative comment
    if st.session_state.qualitative_comment:
        st.markdown("### 💡 Analisi e Suggerimenti")
        st.markdown(f"""
        <div class="brand-card">
            <p style="color: #c8c8d8; font-size: 1.05em; line-height: 1.7; margin: 0;">
                {st.session_state.qualitative_comment}
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    # Detail per question
    st.markdown("### 🔍 Dettagli per Domanda")

    for idx in sorted(st.session_state.eval_results.keys()):
        if idx >= len(QUESTIONS):
            continue
        result = st.session_state.eval_results[idx]
        q = QUESTIONS[idx]
        question_text = q["label"].replace("{BRAND_NAME}", brand_name)
        avg_score = result.get('average_score', 0)
        is_correct = result.get('is_correct', False)

        status_label = "✅ CORRETTA" if is_correct else "❌ DA MIGLIORARE"
        border_color = "#4CAF50" if is_correct else "#F44336"

        with st.expander(f"Domanda {idx + 1}: {question_text[:55]}... -- {status_label}"):
            st.markdown(f"**Domanda:** {question_text}")
            st.markdown(f"**Score medio:** {avg_score:.2f} / 1.00")
            st.markdown("---")

            # Ground truth
            st.markdown("**✅ La tua risposta (Ground Truth):**")
            st.success(st.session_state.user_answers.get(idx, "N/A"))
            st.markdown("---")

            # AI answers
            if idx in st.session_state.ai_answers:
                ai_ans = st.session_state.ai_answers[idx]

                for ai_name, ai_label, ai_icon in [("gemini", "Gemini", "⚫"), ("openai", "ChatGPT", "🟢")]:
                    if ai_name in ai_ans:
                        st.markdown(f"**{ai_icon} {ai_label}:**")
                        st.info(ai_ans[ai_name])
                        if ai_name in result:
                            r = result[ai_name]
                            sc = r.get('score', 0)
                            reason = r.get('reason', '')
                            sc_color = "green" if sc >= 0.75 else ("orange" if sc >= 0.5 else "red")
                            st.markdown(f"Score: :{sc_color}[{sc:.2f}] -- {reason}")
                            if r.get('key_conflicts'):
                                st.markdown(f"Conflitti: {', '.join(r['key_conflicts'])}")
                        st.markdown("")

    st.markdown("---")

    # Recommendation section
    if st.session_state.recommendation:
        st.markdown("### 🏆 Chi consigliano le AI nel tuo settore?")
        st.markdown(f'*Abbiamo chiesto: "Nel settore {sector}, quale brand consiglieresti?"*')

        reco = st.session_state.recommendation

        col1, col2 = st.columns(2)
        with col1:
            if "gemini" in reco:
                mentioned = brand_name.lower() in reco["gemini"].lower()
                border = "#4CAF50" if mentioned else "#FF9800"
                badge = "✅ Ti ha menzionato!" if mentioned else "⚠️ Non ti ha menzionato"
                st.markdown(f"""
                <div class="detail-card" style="border-left-color: {border};">
                    <h4 style="color: white; margin: 0 0 8px;">⚫ Gemini</h4>
                    <p style="color: {border}; font-weight: 700; margin: 0 0 10px;">{badge}</p>
                    <p style="color: #c8c8d8; margin: 0; font-size: 0.95em;">{reco['gemini']}</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            if "openai" in reco:
                mentioned = brand_name.lower() in reco["openai"].lower()
                border = "#4CAF50" if mentioned else "#FF9800"
                badge = "✅ Ti ha menzionato!" if mentioned else "⚠️ Non ti ha menzionato"
                st.markdown(f"""
                <div class="detail-card" style="border-left-color: {border};">
                    <h4 style="color: white; margin: 0 0 8px;">🟢 ChatGPT</h4>
                    <p style="color: {border}; font-weight: 700; margin: 0 0 10px;">{badge}</p>
                    <p style="color: #c8c8d8; margin: 0; font-size: 0.95em;">{reco['openai']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

    # Email section
    st.markdown("### 📧 Ricevi il report completo via email")

    smtp_configured = bool(get_secret("SMTP2GO_USERNAME")) and bool(get_secret("SMTP2GO_PASSWORD"))

    if smtp_configured:
        st.markdown('<div class="email-section">', unsafe_allow_html=True)

        email_input = st.text_input(
            "Il tuo indirizzo email",
            placeholder="nome@azienda.com",
            key="email_input",
        )

        privacy_accepted = st.checkbox(
            "Acconsento al trattamento dei miei dati personali ai sensi della normativa vigente",
            value=True,
            key="privacy_check",
        )

        if st.button("📨 Invia Report", type="primary"):
            if not email_input or "@" not in email_input:
                st.warning("Inserisci un indirizzo email valido")
            elif not privacy_accepted:
                st.warning("Accetta il consenso per procedere")
            else:
                with st.spinner("Generazione PDF e invio in corso..."):
                    pdf = generate_pdf_report(
                        brand_name, sector, summary,
                        st.session_state.eval_results, QUESTIONS,
                        st.session_state.user_answers, st.session_state.ai_answers,
                        st.session_state.recommendation, st.session_state.qualitative_comment,
                    )
                    ok, msg = send_email_report(email_input, brand_name, pdf)
                    if ok:
                        st.success(f"✅ Report inviato a {email_input}!")
                    else:
                        st.error(f"Errore: {msg}")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Fallback: download PDF
        st.markdown("*Invio email non configurato. Puoi scaricare il PDF:*")
        pdf = generate_pdf_report(
            brand_name, sector, summary,
            st.session_state.eval_results, QUESTIONS,
            st.session_state.user_answers, st.session_state.ai_answers,
            st.session_state.recommendation, st.session_state.qualitative_comment,
        )
        st.download_button(
            "📥 Scarica Report PDF",
            data=pdf,
            file_name=f"Brand_AI_Integrity_{brand_name}.pdf",
            mime="application/pdf",
        )

    st.markdown("---")

    # CTA
    st.markdown("""
    <div class="cta-box">
        <h2>🚀 Vuoi migliorare il tuo Brand AI Integrity Score?</h2>
        <p>Il Team Innovation di AvantGrade puo aiutarti a ottimizzare la rappresentazione
        del tuo brand nelle intelligenze artificiali.</p>
        <a href="https://www.avantgrade.com/schedule-a-call">Parliamone insieme →</a>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title="Brand AI Integrity",
        page_icon="🎯",
        layout="centered",
    )

    init_session_state()

    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.title("🎯 Brand AI Integrity")
    st.markdown("**Misura quanto le AI conoscono davvero il tuo brand.**")
    st.markdown("---")

    # Check secrets
    ok, err = check_secrets()
    if not ok:
        st.error(f"⚠️ Configurazione incompleta: {err}")
        st.info("Configura le chiavi API: GEMINI_API_KEY, OPENAI_API_KEY, BRAVE_API_KEY")
        st.stop()

    # Configure models
    gemini_model, openai_client, evaluator_model, err = configure_models()
    if err:
        st.error(err)
        st.stop()

    # Step progress
    current = st.session_state.current_step
    steps = ["Brand & Settore", "Domande & Risposte", "Risultati"]
    st.progress((current - 1) / 2)
    st.markdown(f"**Passo {current}/3:** {steps[current - 1]}")
    st.markdown("---")

    # Render current step
    if current == 1:
        render_step_1()
    elif current == 2:
        render_step_2(gemini_model, openai_client, evaluator_model)
    elif current == 3:
        render_step_3()

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9em;">'
        'Sviluppato dal <b>Team Innovation di AvantGrade.com</b>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔄 Ricomincia", help="Resetta tutto e ricomincia"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
