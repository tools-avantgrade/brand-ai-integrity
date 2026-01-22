"""
Brand AI Integrity Tool - MVP

Misura la Brand Integrity del brand confrontando risposte AI (Gemini, ChatGPT, Claude)
con risposte ground truth fornite dall'utente.
"""

import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from tavily import TavilyClient
import json
import time
from typing import Dict, List, Optional, Tuple


# Configurazione
MATCH_THRESHOLD = 0.75
DEFAULT_QUESTIONS = [
    "Qual è la mission del brand {BRAND_NAME}?",
    "Quali sono i principali prodotti/servizi offerti da {BRAND_NAME}?",
    "Qual è la proposta di valore distintiva di {BRAND_NAME}?",
    "Quali sono 3 punti di forza verificabili di {BRAND_NAME}?",
    "Qual è il pubblico target principale di {BRAND_NAME}?"
]
MIN_QUESTIONS = 3
MAX_QUESTIONS = 10
MIN_ANSWER_LENGTH = 20


def init_session_state():
    """Inizializza lo stato della sessione Streamlit."""
    if 'brand_name' not in st.session_state:
        st.session_state.brand_name = ""
    if 'custom_questions' not in st.session_state:
        st.session_state.custom_questions = []
    if 'ai_answers' not in st.session_state:
        st.session_state.ai_answers = {}
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = {}
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'api_calls_count' not in st.session_state:
        st.session_state.api_calls_count = 0
    if 'last_api_call_time' not in st.session_state:
        st.session_state.last_api_call_time = 0
    if 'use_web_search' not in st.session_state:
        st.session_state.use_web_search = False  # Default: OFF


def get_all_questions():
    """Restituisce tutte le domande (predefinite + personalizzate)."""
    return DEFAULT_QUESTIONS + st.session_state.custom_questions


def check_secrets() -> Tuple[bool, Optional[str]]:
    """Verifica che i secrets necessari siano configurati."""
    try:
        # Verifica Gemini API Key
        gemini_key = st.secrets["GEMINI_API_KEY"]
        if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
            return False, "GEMINI_API_KEY non configurata correttamente in secrets.toml"

        # Verifica OpenAI API Key
        openai_key = st.secrets["OPENAI_API_KEY"]
        if not openai_key or openai_key == "YOUR_OPENAI_API_KEY_HERE":
            return False, "OPENAI_API_KEY non configurata correttamente in secrets.toml"

        # Verifica Anthropic API Key
        anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
        if not anthropic_key or anthropic_key == "YOUR_ANTHROPIC_API_KEY_HERE":
            return False, "ANTHROPIC_API_KEY non configurata correttamente in secrets.toml"

        # Verifica Tavily API Key (per web search)
        tavily_key = st.secrets["TAVILY_API_KEY"]
        if not tavily_key or tavily_key == "YOUR_TAVILY_API_KEY_HERE":
            return False, "TAVILY_API_KEY non configurata correttamente in secrets.toml"

        return True, None
    except KeyError as e:
        return False, f"Chiave API mancante in secrets.toml: {str(e)}"
    except Exception as e:
        return False, f"Errore nel leggere secrets: {str(e)}"


def configure_ai_models() -> Tuple[Optional[genai.GenerativeModel], Optional[OpenAI], Optional[Anthropic], Optional[genai.GenerativeModel], Optional[str]]:
    """Configura tutti i modelli AI (Gemini, OpenAI, Claude) e l'evaluator."""
    try:
        # Configura Gemini
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=gemini_api_key)

        gemini_model_name = st.secrets.get("GEMINI_MODEL", "gemini-3-flash-preview")
        evaluator_model_name = st.secrets.get("EVALUATOR_MODEL", gemini_model_name)

        # Config per modello di generazione
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

        # Config per evaluator: più token e response JSON mode
        evaluator_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }

        gemini_model = genai.GenerativeModel(
            model_name=gemini_model_name,
            generation_config=generation_config
        )

        evaluator_model = genai.GenerativeModel(
            model_name=evaluator_model_name,
            generation_config=evaluator_config
        )

        # Configura OpenAI
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # Configura Anthropic (Claude)
        anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

        return gemini_model, openai_client, anthropic_client, evaluator_model, None
    except Exception as e:
        return None, None, None, None, f"Errore nella configurazione AI: {str(e)}"


def rate_limit_check():
    """Semplice rate limiting in-memory per sessione."""
    current_time = time.time()
    time_window = 60  # 1 minuto
    max_calls = 30  # max 30 chiamate al minuto

    # Reset contatore se è passato un minuto
    if current_time - st.session_state.last_api_call_time > time_window:
        st.session_state.api_calls_count = 0
        st.session_state.last_api_call_time = current_time

    # Verifica limite
    if st.session_state.api_calls_count >= max_calls:
        remaining_time = int(time_window - (current_time - st.session_state.last_api_call_time))
        return False, f"Limite rate (30 chiamate/min) raggiunto. Riprova tra {remaining_time}s"

    # Incrementa contatore
    st.session_state.api_calls_count += 1
    return True, None


@st.cache_data(ttl=600, show_spinner=False)  # Cache 10 minuti
def web_search(query: str, max_results: int = 5) -> Tuple[str, bool]:
    """
    Effettua una ricerca web usando Tavily (molto più affidabile di DuckDuckGo).

    Args:
        query: Query di ricerca
        max_results: Numero massimo di risultati da restituire

    Returns:
        Tupla (risultati formattati, successo)
    """
    try:
        tavily_key = st.secrets.get("TAVILY_API_KEY", "")
        if not tavily_key:
            return "Tavily API key non configurata.", False

        tavily_client = TavilyClient(api_key=tavily_key)

        # Esegui ricerca con Tavily
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",  # Ricerca approfondita
            include_answer=True,  # Include risposta diretta
            include_raw_content=False
        )

        if not response or 'results' not in response or not response['results']:
            return "Nessun risultato trovato dalla ricerca web.", False

        # Formatta i risultati
        formatted_results = ""

        # Aggiungi risposta diretta se disponibile
        if response.get('answer'):
            formatted_results += f"**Risposta diretta:**\n{response['answer']}\n\n"

        formatted_results += "**Fonti verificate:**\n\n"

        for idx, result in enumerate(response['results'], 1):
            title = result.get('title', 'N/A')
            content = result.get('content', 'N/A')
            url = result.get('url', '')
            score = result.get('score', 0)

            formatted_results += f"{idx}. **{title}** (relevance: {score:.2f})\n{content}\nURL: {url}\n\n"

        return formatted_results.strip(), True

    except Exception as e:
        return f"Errore durante la ricerca web: {str(e)}", False


@st.cache_data(ttl=600, show_spinner=False)  # Cache ridotta
def generate_gemini_answer(_model: genai.GenerativeModel, brand_name: str, question: str) -> Tuple[Optional[str], Optional[str]]:
    """Genera risposta da Gemini con accesso web nativo."""
    try:
        final_question = question.replace("{BRAND_NAME}", brand_name)
        prompt = f"""Rispondi alla seguente domanda su {brand_name} utilizzando informazioni aggiornate e verificate che puoi trovare online.

Domanda: {final_question}

IMPORTANTE:
- Cerca attivamente informazioni recenti sul web riguardo a {brand_name}
- Se non trovi informazioni specifiche sul sito ufficiale o fonti verificate, usa la tua conoscenza generale ma specifica che potrebbe non essere aggiornata
- Non dire semplicemente "non ho trovato informazioni" - fornisci comunque una risposta basata sulla tua conoscenza

Rispondi in italiano, in modo chiaro e diretto (massimo 200 parole)."""

        response = _model.generate_content(prompt)
        if response and response.text:
            return response.text.strip(), None
        else:
            return None, "Risposta vuota da Gemini"
    except Exception as e:
        return None, f"Errore Gemini: {str(e)}"


@st.cache_data(ttl=600, show_spinner=False)  # Cache ridotta
def generate_openai_answer(_client: OpenAI, brand_name: str, question: str) -> Tuple[Optional[str], Optional[str]]:
    """Genera risposta da ChatGPT con ricerca web Tavily."""
    try:
        final_question = question.replace("{BRAND_NAME}", brand_name)
        openai_model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

        # Ricerca web con Tavily (molto più affidabile)
        search_query = f"{brand_name} {final_question}"
        search_results, search_success = web_search(search_query, max_results=5)

        # Costruisci il prompt basandoti sul successo della ricerca
        if search_success:
            system_prompt = "Sei un assistente esperto che risponde basandosi ESCLUSIVAMENTE su informazioni verificate dalla ricerca web fornita. Usa SOLO le informazioni trovate nelle fonti. Rispondi in modo dettagliato e preciso."
            user_prompt = f"""Domanda: {final_question}

INFORMAZIONI DALLA RICERCA WEB (USA SOLO QUESTE):
{search_results}

Rispondi alla domanda in italiano basandoti ESCLUSIVAMENTE sulle informazioni sopra (massimo 200 parole).
Cita le fonti quando possibile."""
        else:
            system_prompt = "Sei un assistente esperto che risponde basandosi sulla tua conoscenza interna quando la ricerca web fallisce."
            user_prompt = f"""Domanda: {final_question}

La ricerca web non ha prodotto risultati utili. Rispondi basandoti sulla tua conoscenza generale del brand {brand_name}.
Rispondi in italiano, in modo chiaro (massimo 200 parole). Specifica che le informazioni potrebbero non essere le più aggiornate."""

        response = _client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )

        if response and response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip(), None
        else:
            return None, "Risposta vuota da ChatGPT"
    except Exception as e:
        return None, f"Errore ChatGPT: {str(e)}"


@st.cache_data(ttl=600, show_spinner=False)  # Cache ridotta
def generate_claude_answer(_client: Anthropic, brand_name: str, question: str) -> Tuple[Optional[str], Optional[str]]:
    """Genera risposta da Claude con ricerca web Tavily."""
    try:
        final_question = question.replace("{BRAND_NAME}", brand_name)
        claude_model = st.secrets.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

        # Ricerca web con Tavily (molto più affidabile)
        search_query = f"{brand_name} {final_question}"
        search_results, search_success = web_search(search_query, max_results=5)

        # Costruisci il prompt basandoti sul successo della ricerca
        if search_success:
            user_prompt = f"""Rispondi alla seguente domanda basandoti ESCLUSIVAMENTE sulle informazioni verificate dalla ricerca web.

Domanda: {final_question}

INFORMAZIONI DALLA RICERCA WEB (USA SOLO QUESTE):
{search_results}

Rispondi alla domanda in italiano basandoti ESCLUSIVAMENTE sulle informazioni sopra (massimo 200 parole).
Cita le fonti quando possibile e fornisci una risposta dettagliata e precisa."""
        else:
            user_prompt = f"""Rispondi alla seguente domanda basandoti sulla tua conoscenza interna.

Domanda: {final_question}

La ricerca web non ha prodotto risultati utili per il brand {brand_name}.
Rispondi basandoti sulla tua conoscenza generale in italiano, in modo chiaro (massimo 200 parole).
Specifica che le informazioni potrebbero non essere le più aggiornate."""

        response = _client.messages.create(
            model=claude_model,
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        if response and response.content and len(response.content) > 0:
            return response.content[0].text.strip(), None
        else:
            return None, "Risposta vuota da Claude"
    except Exception as e:
        return None, f"Errore Claude: {str(e)}"


def evaluate_answer(_model: genai.GenerativeModel, question: str, ai_answer: str, user_answer: str, retry: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Valuta la coerenza tra risposta AI e risposta utente.
    Restituisce dict con score, is_correct, reason, key_conflicts.
    """
    try:
        prompt = f"""Valuta la coerenza tra la risposta AI e la risposta ground truth (utente).
Le risposte possono contenere elenchi puntati o testo su più righe.

Domanda: {question}

Risposta AI:
{ai_answer}

Risposta ground truth (utente):
{user_answer}

Criteri di valutazione:
- "corretta" (score >= 0.75) se semanticamente allineata alla ground truth e non contraddice
- "sbagliata" (score < 0.75) se contraddice, oppure aggiunge affermazioni specifiche incompatibili, oppure manca elementi essenziali quando la ground truth li indica chiaramente

{"IMPORTANTE: Genera SOLO JSON valido. Il campo 'reason' deve essere una SINGOLA frase breve (max 100 caratteri)." if retry else ""}

Restituisci un oggetto JSON con la seguente struttura:
{{
  "score": 0.85,
  "is_correct": true,
  "reason": "Breve spiegazione in una frase",
  "key_conflicts": ["conflitto 1", "conflitto 2"]
}}

Schema JSON:
- score: numero decimale da 0.0 a 1.0 (allineamento semantico)
- is_correct: booleano true se score >= {MATCH_THRESHOLD}, altrimenti false
- reason: stringa breve (max 100 caratteri, 1 frase senza a capo)
- key_conflicts: array di stringhe (max 3 elementi, può essere vuoto [])
"""

        response = _model.generate_content(prompt)

        if not response or not response.text:
            return None, "Risposta vuota dall'evaluator"

        # Parsing JSON robusto
        response_text = response.text.strip()

        # Rimuovi markdown code blocks se presenti
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        # Rimuovi eventuali spazi bianchi o caratteri nascosti
        response_text = response_text.strip()

        try:
            result = json.loads(response_text)

            # Validazione schema
            required_fields = ["score", "is_correct", "reason"]
            for field in required_fields:
                if field not in result:
                    return None, f"Campo mancante nel JSON: {field}"

            # Normalizza score e is_correct
            result["score"] = float(result["score"])
            result["is_correct"] = bool(result["is_correct"])

            # Normalizza reason: rimuovi newline se presenti
            if isinstance(result["reason"], str):
                result["reason"] = result["reason"].replace("\n", " ").strip()

            # key_conflicts opzionale
            if "key_conflicts" not in result:
                result["key_conflicts"] = []

            return result, None

        except json.JSONDecodeError as e:
            # Retry una volta
            if not retry:
                return evaluate_answer(_model, question, ai_answer, user_answer, retry=True)
            return None, f"Errore parsing JSON: {str(e)}\nRisposta: {response_text[:300]}"

    except Exception as e:
        return None, f"Errore valutazione: {str(e)}"


def render_section_a():
    """Sezione A: Setup - Brand name, domande e risposte."""
    st.header("Sezione A: Setup e Risposte")

    # Brand name
    brand_name = st.text_input(
        "Nome del Brand",
        value=st.session_state.brand_name,
        placeholder="es. Nike, Apple, Tesla...",
        help="Inserisci il nome del brand da analizzare"
    )

    if brand_name != st.session_state.brand_name:
        st.session_state.brand_name = brand_name
        # Reset quando cambia brand
        st.session_state.ai_answers = {}
        st.session_state.user_answers = {}
        st.session_state.eval_results = {}
        st.session_state.summary = None

    if not brand_name:
        st.warning("Inserisci il nome del brand per procedere")
        return False

    # Domande e risposte
    st.subheader("Domande e Risposte")
    st.markdown("**Domande predefinite** (non modificabili):")

    all_valid = True

    # Domande predefinite con risposte inline
    for idx, q in enumerate(DEFAULT_QUESTIONS):
        # Mostra domanda con brand name sostituito
        question = q.replace("{BRAND_NAME}", brand_name)

        st.text_area(
            f"Domanda {idx + 1}",
            value=question,
            key=f"default_question_{idx}",
            height=60,
            disabled=True,
            help="Domanda predefinita (non modificabile)"
        )

        # Risposta utente subito sotto
        user_answer = st.text_area(
            f"Risposta alla domanda {idx + 1}",
            value=st.session_state.user_answers.get(idx, ""),
            key=f"user_answer_{idx}",
            height=120,
            placeholder="Inserisci qui la risposta corretta del brand...",
            help="Inserisci la risposta corretta secondo il brand (min 20 caratteri)"
        )

        st.session_state.user_answers[idx] = user_answer

        # Validazione lunghezza
        if len(user_answer.strip()) < MIN_ANSWER_LENGTH:
            st.warning(f"⚠️ Risposta troppo corta (min {MIN_ANSWER_LENGTH} caratteri)")
            all_valid = False

        st.divider()

    # Domande personalizzate (modificabili) con risposte inline
    custom_questions = st.session_state.custom_questions

    if custom_questions:
        st.markdown("**Domande personalizzate:**")

    for idx, q in enumerate(custom_questions):
        question_idx = len(DEFAULT_QUESTIONS) + idx

        col1, col2 = st.columns([5, 1])
        with col1:
            # Mostra preview con brand name sostituito
            preview = q.replace("{BRAND_NAME}", brand_name)

            new_q = st.text_area(
                f"Domanda personalizzata {idx + 1}",
                value=q,
                key=f"custom_question_{idx}",
                height=60,
                help=f"Preview: {preview}"
            )
            if new_q != q:
                st.session_state.custom_questions[idx] = new_q

        with col2:
            st.write("")  # spacing
            st.write("")  # spacing
            if st.button("Rimuovi", key=f"remove_custom_{idx}"):
                st.session_state.custom_questions.pop(idx)
                # Rimuovi anche la risposta corrispondente
                if question_idx in st.session_state.user_answers:
                    del st.session_state.user_answers[question_idx]
                st.rerun()

        # Risposta utente subito sotto
        user_answer = st.text_area(
            f"Risposta alla domanda personalizzata {idx + 1}",
            value=st.session_state.user_answers.get(question_idx, ""),
            key=f"user_answer_{question_idx}",
            height=120,
            placeholder="Inserisci qui la risposta corretta del brand...",
            help="Inserisci la risposta corretta secondo il brand (min 20 caratteri)"
        )

        st.session_state.user_answers[question_idx] = user_answer

        # Validazione lunghezza
        if len(user_answer.strip()) < MIN_ANSWER_LENGTH:
            st.warning(f"⚠️ Risposta troppo corta (min {MIN_ANSWER_LENGTH} caratteri)")
            all_valid = False

        st.divider()

    # Pulsante aggiungi domanda personalizzata
    all_questions = get_all_questions()
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("➕ Aggiungi domanda personalizzata", disabled=len(all_questions) >= MAX_QUESTIONS):
            st.session_state.custom_questions.append(f"Nuova domanda su {{BRAND_NAME}}?")
            st.rerun()

    with col2:
        st.caption(f"Domande totali: {len(all_questions)} ({len(DEFAULT_QUESTIONS)} predefinite + {len(custom_questions)} personalizzate, max {MAX_QUESTIONS})")

    return all_valid


def render_section_b(gemini_model: genai.GenerativeModel, openai_client: OpenAI, anthropic_client: Anthropic):
    """Sezione B: Generazione risposte AI (Gemini, ChatGPT, Claude)."""
    st.header("Sezione B: Risposte AI")

    # Verifica che le risposte utente siano state inserite
    questions = get_all_questions()

    # Verifica che tutte le risposte siano valide
    all_user_answers_valid = all(
        len(st.session_state.user_answers.get(idx, "").strip()) >= MIN_ANSWER_LENGTH
        for idx in range(len(questions))
    )

    if not all_user_answers_valid:
        st.info("Completa prima le risposte ground truth nella Sezione A")
        return False

    brand_name = st.session_state.brand_name

    # Pulsante genera
    if st.button("🌐 Genera risposte AI (Web Search + Conoscenza Interna)", type="primary", disabled=not brand_name):
        # Rate limiting
        can_proceed, error_msg = rate_limit_check()
        if not can_proceed:
            st.error(error_msg)
            return False

        progress_bar = st.progress(0)
        status_text = st.empty()

        st.session_state.ai_answers = {}
        errors = []

        total_steps = len(questions) * 3  # 3 AI per domanda
        current_step = 0

        for idx, question in enumerate(questions):
            st.session_state.ai_answers[idx] = {}

            # Genera risposta Gemini
            status_text.text(f"Generando risposta Gemini per domanda {idx + 1}/{len(questions)}...")
            progress_bar.progress(current_step / total_steps)

            gemini_answer, gemini_error = generate_gemini_answer(gemini_model, brand_name, question)
            if gemini_error:
                errors.append(f"Domanda {idx + 1} (Gemini): {gemini_error}")
            else:
                st.session_state.ai_answers[idx]["gemini"] = gemini_answer
            current_step += 1

            # Genera risposta ChatGPT
            status_text.text(f"Generando risposta ChatGPT per domanda {idx + 1}/{len(questions)}...")
            progress_bar.progress(current_step / total_steps)

            openai_answer, openai_error = generate_openai_answer(openai_client, brand_name, question)
            if openai_error:
                errors.append(f"Domanda {idx + 1} (ChatGPT): {openai_error}")
            else:
                st.session_state.ai_answers[idx]["openai"] = openai_answer
            current_step += 1

            # Genera risposta Claude
            status_text.text(f"Generando risposta Claude per domanda {idx + 1}/{len(questions)}...")
            progress_bar.progress(current_step / total_steps)

            claude_answer, claude_error = generate_claude_answer(anthropic_client, brand_name, question)
            if claude_error:
                errors.append(f"Domanda {idx + 1} (Claude): {claude_error}")
            else:
                st.session_state.ai_answers[idx]["claude"] = claude_answer
            current_step += 1

            # Rate limiting check
            can_proceed, error_msg = rate_limit_check()
            if not can_proceed:
                st.error(error_msg)
                break

        progress_bar.progress(1.0)
        status_text.text("Completato!")

        if errors:
            st.error("Alcuni errori durante la generazione:")
            for err in errors:
                st.text(err)
        else:
            total_responses = sum(len(answers) for answers in st.session_state.ai_answers.values())
            st.success(f"Generate {total_responses} risposte AI ({len(questions)} domande × 3 AI)")

        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

    # Mostra risposte AI generate
    if st.session_state.ai_answers:
        st.subheader("Risposte AI generate")

        for idx, question in enumerate(questions):
            if idx in st.session_state.ai_answers:
                with st.expander(f"Domanda {idx + 1}: {question.replace('{BRAND_NAME}', brand_name)[:60]}..."):
                    st.markdown(f"**Domanda:** {question.replace('{BRAND_NAME}', brand_name)}")

                    # Mostra le 3 risposte in colonne
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**🤖 Gemini**")
                        if "gemini" in st.session_state.ai_answers[idx]:
                            st.info(st.session_state.ai_answers[idx]["gemini"])
                        else:
                            st.warning("Non disponibile")

                    with col2:
                        st.markdown("**💬 ChatGPT**")
                        if "openai" in st.session_state.ai_answers[idx]:
                            st.info(st.session_state.ai_answers[idx]["openai"])
                        else:
                            st.warning("Non disponibile")

                    with col3:
                        st.markdown("**🧠 Claude**")
                        if "claude" in st.session_state.ai_answers[idx]:
                            st.info(st.session_state.ai_answers[idx]["claude"])
                        else:
                            st.warning("Non disponibile")

    return len(st.session_state.ai_answers) > 0


def render_section_c(evaluator_model: genai.GenerativeModel):
    """Sezione C: Calcolo Brand Integrity."""
    st.header("Sezione C: Calcolo Brand Integrity")

    # Verifica prerequisiti
    if not st.session_state.user_answers:
        st.info("Inserisci le risposte ground truth nella Sezione A")
        return False

    if not st.session_state.ai_answers:
        st.info("Genera prima le risposte AI nella Sezione B")
        return False

    questions = get_all_questions()

    # Verifica che tutte le risposte siano valide
    all_valid = all(
        len(st.session_state.user_answers.get(idx, "").strip()) >= MIN_ANSWER_LENGTH
        for idx in st.session_state.ai_answers.keys()
    )

    if not all_valid:
        st.warning("Completa tutte le risposte ground truth (min 20 caratteri ciascuna)")
        return False

    # Pulsante calcola
    if st.button("Calcola Brand Integrity (3 AI)", type="primary"):
        # Rate limiting
        can_proceed, error_msg = rate_limit_check()
        if not can_proceed:
            st.error(error_msg)
            return False

        progress_bar = st.progress(0)
        status_text = st.empty()

        st.session_state.eval_results = {}
        errors = []

        brand_name = st.session_state.brand_name
        ai_models = ["gemini", "openai", "claude"]
        total_evals = len(st.session_state.ai_answers) * len(ai_models)
        current_eval = 0

        for idx in sorted(st.session_state.ai_answers.keys()):
            question = questions[idx].replace("{BRAND_NAME}", brand_name)
            ai_answers = st.session_state.ai_answers[idx]
            user_answer = st.session_state.user_answers[idx]

            st.session_state.eval_results[idx] = {}
            scores = []

            # Valuta ciascuna AI
            for ai_name in ai_models:
                if ai_name in ai_answers:
                    status_text.text(f"Valutando {ai_name.capitalize()} per domanda {idx + 1}/{len(st.session_state.ai_answers)}...")
                    progress_bar.progress(current_eval / total_evals)

                    result, error = evaluate_answer(evaluator_model, question, ai_answers[ai_name], user_answer)

                    if error:
                        errors.append(f"Domanda {idx + 1} ({ai_name.capitalize()}): {error}")
                    else:
                        st.session_state.eval_results[idx][ai_name] = result
                        scores.append(result['score'])

                    current_eval += 1

                    # Rate limiting check
                    can_proceed, error_msg = rate_limit_check()
                    if not can_proceed:
                        st.error(error_msg)
                        break

            # Calcola media score per questa domanda
            if scores:
                avg_score = sum(scores) / len(scores)
                st.session_state.eval_results[idx]['average_score'] = avg_score
                st.session_state.eval_results[idx]['is_correct'] = avg_score >= MATCH_THRESHOLD

        progress_bar.progress(1.0)
        status_text.text("Valutazione completata!")

        # Calcola summary
        if st.session_state.eval_results:
            total = len(st.session_state.eval_results)
            correct = sum(1 for r in st.session_state.eval_results.values() if r.get('is_correct', False))
            integrity_score = round((correct / total) * 100)

            # Calcola score medi per ogni AI
            ai_scores = {ai: [] for ai in ai_models}
            for result in st.session_state.eval_results.values():
                for ai_name in ai_models:
                    if ai_name in result and 'score' in result[ai_name]:
                        ai_scores[ai_name].append(result[ai_name]['score'])

            ai_averages = {
                ai: round(sum(scores) / len(scores) * 100) if scores else 0
                for ai, scores in ai_scores.items()
            }

            st.session_state.summary = {
                'total': total,
                'correct': correct,
                'incorrect': total - correct,
                'integrity_score': integrity_score,
                'ai_scores': ai_averages
            }

        if errors:
            st.error("Alcuni errori durante la valutazione:")
            for err in errors:
                st.text(err)
        else:
            st.success("Valutazione completata!")

        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

        return True

    return st.session_state.summary is not None


def render_section_d():
    """Sezione D: Visualizzazione risultati."""
    st.header("Sezione D: Risultati")

    if not st.session_state.summary:
        st.info("Calcola prima il Brand Integrity Score nella Sezione C")
        return

    summary = st.session_state.summary

    # Score grande (media di tutte le AI)
    st.markdown("### Brand Integrity Score (Media)")

    # Colore basato su score
    score = summary['integrity_score']
    if score >= 80:
        color = "green"
    elif score >= 60:
        color = "orange"
    else:
        color = "red"

    st.markdown(f"<h1 style='text-align: center; color: {color};'>{score}/100</h1>", unsafe_allow_html=True)

    # Statistiche generali
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Totale domande", summary['total'])
    with col2:
        st.metric("Corrette (avg)", summary['correct'])
    with col3:
        st.metric("Sbagliate (avg)", summary['incorrect'])

    # Score individuali per AI
    st.markdown("### Score per AI")
    col1, col2, col3 = st.columns(3)

    ai_scores = summary.get('ai_scores', {})

    with col1:
        gemini_score = ai_scores.get('gemini', 0)
        st.metric("🤖 Gemini", f"{gemini_score}/100")

    with col2:
        openai_score = ai_scores.get('openai', 0)
        st.metric("💬 ChatGPT", f"{openai_score}/100")

    with col3:
        claude_score = ai_scores.get('claude', 0)
        st.metric("🧠 Claude", f"{claude_score}/100")

    st.divider()

    # Tabella dettagli
    st.subheader("Dettagli per domanda")

    questions = get_all_questions()
    brand_name = st.session_state.brand_name

    for idx in sorted(st.session_state.eval_results.keys()):
        result = st.session_state.eval_results[idx]
        question = questions[idx].replace("{BRAND_NAME}", brand_name)

        # Colore status basato su media
        if result.get('is_correct', False):
            status = "CORRETTA"
            status_color = "green"
        else:
            status = "SBAGLIATA"
            status_color = "red"

        avg_score = result.get('average_score', 0)

        # Expander per ogni domanda
        with st.expander(f"Domanda {idx + 1}: {question[:60]}... - {status} (avg: {avg_score:.2f})"):
            st.markdown(f"**Domanda completa:** {question}")
            st.markdown(f"**Esito:** :{status_color}[{status}]")
            st.markdown(f"**Score Medio:** {avg_score:.2f} / 1.00")

            # Dettagli per ogni AI
            st.markdown("---")
            st.markdown("**Valutazioni per AI:**")

            for ai_name, ai_label, ai_icon in [
                ("gemini", "Gemini", "🤖"),
                ("openai", "ChatGPT", "💬"),
                ("claude", "Claude", "🧠")
            ]:
                if ai_name in result:
                    ai_result = result[ai_name]
                    ai_status_color = "green" if ai_result.get('is_correct', False) else "red"

                    st.markdown(f"**{ai_icon} {ai_label}**")
                    st.markdown(f"- Esito: :{ai_status_color}[{'CORRETTA' if ai_result.get('is_correct') else 'SBAGLIATA'}]")
                    st.markdown(f"- Score: {ai_result.get('score', 0):.2f} / 1.00")
                    st.markdown(f"- Motivazione: {ai_result.get('reason', 'N/A')}")

                    if ai_result.get('key_conflicts'):
                        st.markdown(f"- Conflitti: {', '.join(ai_result['key_conflicts'])}")

                    st.markdown("")

            # Mostra risposte confrontate
            st.markdown("---")
            st.markdown("**Confronto risposte:**")

            # Ground truth
            st.markdown("**✅ Risposta Ground Truth (Utente):**")
            st.success(st.session_state.user_answers[idx])

            # Risposte AI
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🤖 Gemini:**")
                if "gemini" in st.session_state.ai_answers[idx]:
                    st.info(st.session_state.ai_answers[idx]["gemini"])
                else:
                    st.warning("Non disponibile")

            with col2:
                st.markdown("**💬 ChatGPT:**")
                if "openai" in st.session_state.ai_answers[idx]:
                    st.info(st.session_state.ai_answers[idx]["openai"])
                else:
                    st.warning("Non disponibile")

            with col3:
                st.markdown("**🧠 Claude:**")
                if "claude" in st.session_state.ai_answers[idx]:
                    st.info(st.session_state.ai_answers[idx]["claude"])
                else:
                    st.warning("Non disponibile")


def main():
    """Main app."""
    st.set_page_config(
        page_title="Brand AI Integrity Tool",
        page_icon="🎯",
        layout="wide"
    )

    st.title("Brand AI Integrity Tool")
    st.markdown("Misura la Brand Integrity confrontando risposte AI (Gemini, ChatGPT, Claude) con risposte ground truth del brand.")
    st.success("🌐 **Ricerca Web Avanzata con Tavily API** - Le AI usano fonti verificate dal web in tempo reale!")

    # Init session state
    init_session_state()

    # Check secrets
    secrets_ok, error_msg = check_secrets()
    if not secrets_ok:
        st.error(f"Errore configurazione: {error_msg}")
        st.info("""Configura il file .streamlit/secrets.toml con le chiavi API:
        - GEMINI_API_KEY
        - OPENAI_API_KEY
        - ANTHROPIC_API_KEY
        - TAVILY_API_KEY (gratuita su tavily.com - 1000 ricerche/mese)
        """)
        st.stop()

    # Configure AI models
    gemini_model, openai_client, anthropic_client, evaluator_model, error_msg = configure_ai_models()
    if error_msg:
        st.error(error_msg)
        st.stop()

    # Sidebar info
    with st.sidebar:
        st.header("Informazioni")
        st.markdown(f"""
        **🤖 Gemini:** {st.secrets.get('GEMINI_MODEL', 'gemini-3-flash-preview')}
        🌐 *Accesso web nativo Google*

        **💬 ChatGPT:** {st.secrets.get('OPENAI_MODEL', 'gpt-4o-mini')}
        🔍 *Tavily Web Search API*

        **🧠 Claude:** {st.secrets.get('CLAUDE_MODEL', 'claude-sonnet-4-5-20250929')}
        🔍 *Tavily Web Search API*

        **Evaluator:** {st.secrets.get('EVALUATOR_MODEL', 'gemini-3-flash-preview')}

        **Soglia match:** {MATCH_THRESHOLD}

        **Chiamate API (sessione):** {st.session_state.api_calls_count}

        ---

        🔍 **Tavily API**: Ricerca web professionale con fonti verificate e scoring di rilevanza. Molto più accurata di DuckDuckGo!
        """)

        if st.button("Reset sessione"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Sezioni dell'app
    st.divider()

    # Sezione A: Setup + Risposte (unificata)
    user_answers_ready = render_section_a()

    st.divider()

    # Sezione B: Generazione risposte AI (3 AI)
    if user_answers_ready:
        ai_answers_ready = render_section_b(gemini_model, openai_client, anthropic_client)
    else:
        st.info("Completa le risposte nella Sezione A per procedere")
        ai_answers_ready = False

    st.divider()

    # Sezione C: Calcolo Brand Integrity (3 AI)
    if ai_answers_ready:
        calculation_done = render_section_c(evaluator_model)
    else:
        st.info("Completa le sezioni precedenti per calcolare il Brand Integrity Score")
        calculation_done = False

    st.divider()

    # Sezione D: Risultati (3 AI)
    render_section_d()

    # Footer
    st.divider()
    st.caption("Brand AI Integrity Tool - MVP v1.0")


if __name__ == "__main__":
    main()
