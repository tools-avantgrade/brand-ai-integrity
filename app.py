"""
Brand AI Integrity Tool - MVP

Misura la Brand Integrity del brand confrontando risposte AI (Gemini)
con risposte ground truth fornite dall'utente.
"""

import streamlit as st
import google.generativeai as genai
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
    if 'questions' not in st.session_state:
        st.session_state.questions = DEFAULT_QUESTIONS.copy()
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


def check_secrets() -> Tuple[bool, Optional[str]]:
    """Verifica che i secrets necessari siano configurati."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            return False, "GEMINI_API_KEY non configurata correttamente in secrets.toml"
        return True, None
    except KeyError:
        return False, "GEMINI_API_KEY mancante in secrets.toml"
    except Exception as e:
        return False, f"Errore nel leggere secrets: {str(e)}"


def configure_gemini() -> Tuple[Optional[genai.GenerativeModel], Optional[genai.GenerativeModel], Optional[str]]:
    """Configura i modelli Gemini per generazione e valutazione."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)

        # Leggi i nomi dei modelli dai secrets con fallback
        gemini_model_name = st.secrets.get("GEMINI_MODEL", "gemini-3-flash-preview")
        evaluator_model_name = st.secrets.get("EVALUATOR_MODEL", gemini_model_name)

        # Configura i modelli con generation_config per stabilità
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

        gemini_model = genai.GenerativeModel(
            model_name=gemini_model_name,
            generation_config=generation_config
        )

        evaluator_model = genai.GenerativeModel(
            model_name=evaluator_model_name,
            generation_config=generation_config
        )

        return gemini_model, evaluator_model, None
    except Exception as e:
        return None, None, f"Errore nella configurazione Gemini: {str(e)}"


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


@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_answer(_model: genai.GenerativeModel, brand_name: str, question: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Genera risposta AI per una domanda sul brand.
    Usa caching per evitare chiamate duplicate.
    """
    try:
        # Sostituisci placeholder brand name
        final_question = question.replace("{BRAND_NAME}", brand_name)

        # Prompt con istruzioni per risposte prudenti
        prompt = f"""Rispondi alla seguente domanda in modo conciso e prudente.
Non inventare dettagli specifici (numeri, date, paesi, certificazioni) se non sei sicuro.
Se manca informazione, dichiaralo esplicitamente.

Domanda: {final_question}

Rispondi in italiano, in modo chiaro e diretto (massimo 200 parole)."""

        response = _model.generate_content(prompt)

        if response and response.text:
            return response.text.strip(), None
        else:
            return None, "Risposta vuota dal modello"

    except Exception as e:
        return None, f"Errore generazione risposta: {str(e)}"


def evaluate_answer(_model: genai.GenerativeModel, question: str, ai_answer: str, user_answer: str, retry: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Valuta la coerenza tra risposta AI e risposta utente.
    Restituisce dict con score, is_correct, reason, key_conflicts.
    """
    try:
        prompt = f"""Valuta la coerenza tra la risposta AI e la risposta ground truth (utente).

Domanda: {question}

Risposta AI:
{ai_answer}

Risposta ground truth (utente):
{user_answer}

Criteri di valutazione:
- "corretta" (score >= 0.75) se semanticamente allineata alla ground truth e non contraddice
- "sbagliata" (score < 0.75) se contraddice, oppure aggiunge affermazioni specifiche incompatibili, oppure manca elementi essenziali quando la ground truth li indica chiaramente

{"ATTENZIONE: Rispondi SOLO con JSON valido, senza testo aggiuntivo prima o dopo." if retry else ""}

Rispondi SOLO con un oggetto JSON nel seguente formato (niente altro testo):
{{
  "score": 0.85,
  "is_correct": true,
  "reason": "La risposta AI è semanticamente allineata...",
  "key_conflicts": ["eventuale conflitto 1", "conflitto 2"]
}}

Dove:
- score: float da 0.0 a 1.0 (allineamento semantico)
- is_correct: true se score >= {MATCH_THRESHOLD}, altrimenti false
- reason: spiegazione breve (1-2 frasi)
- key_conflicts: array di stringhe (max 3, può essere vuoto)
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

            # key_conflicts opzionale
            if "key_conflicts" not in result:
                result["key_conflicts"] = []

            return result, None

        except json.JSONDecodeError as e:
            # Retry una volta
            if not retry:
                return evaluate_answer(_model, question, ai_answer, user_answer, retry=True)
            return None, f"Errore parsing JSON: {str(e)}\nRisposta: {response_text[:200]}"

    except Exception as e:
        return None, f"Errore valutazione: {str(e)}"


def render_section_a():
    """Sezione A: Setup - Brand name e domande."""
    st.header("Sezione A: Setup")

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

    # Editor domande
    st.subheader("Domande da porre")

    questions = st.session_state.questions

    # Mostra domande esistenti
    for idx, q in enumerate(questions):
        col1, col2 = st.columns([5, 1])
        with col1:
            # Mostra preview con brand name sostituito
            if brand_name:
                preview = q.replace("{BRAND_NAME}", brand_name)
            else:
                preview = q

            new_q = st.text_area(
                f"Domanda {idx + 1}",
                value=q,
                key=f"question_{idx}",
                height=80,
                help=f"Preview: {preview}"
            )
            if new_q != q:
                st.session_state.questions[idx] = new_q

        with col2:
            st.write("")  # spacing
            st.write("")  # spacing
            if st.button("Rimuovi", key=f"remove_{idx}", disabled=len(questions) <= MIN_QUESTIONS):
                st.session_state.questions.pop(idx)
                st.rerun()

    # Pulsante aggiungi domanda
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Aggiungi domanda", disabled=len(questions) >= MAX_QUESTIONS):
            st.session_state.questions.append(f"Nuova domanda su {{BRAND_NAME}}?")
            st.rerun()

    with col2:
        st.caption(f"Domande: {len(questions)} (min {MIN_QUESTIONS}, max {MAX_QUESTIONS})")

    # Validazione
    if not brand_name:
        st.warning("Inserisci il nome del brand per procedere")
        return False

    if len([q for q in questions if q.strip()]) < MIN_QUESTIONS:
        st.warning(f"Inserisci almeno {MIN_QUESTIONS} domande")
        return False

    return True


def render_section_b(gemini_model: genai.GenerativeModel):
    """Sezione B: Generazione risposte AI."""
    st.header("Sezione B: Risposte AI")

    brand_name = st.session_state.brand_name
    questions = st.session_state.questions

    # Pulsante genera
    if st.button("Genera risposte AI", type="primary", disabled=not brand_name):
        # Rate limiting
        can_proceed, error_msg = rate_limit_check()
        if not can_proceed:
            st.error(error_msg)
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        st.session_state.ai_answers = {}
        errors = []

        for idx, question in enumerate(questions):
            status_text.text(f"Generando risposta {idx + 1}/{len(questions)}...")
            progress_bar.progress((idx) / len(questions))

            answer, error = generate_ai_answer(gemini_model, brand_name, question)

            if error:
                errors.append(f"Domanda {idx + 1}: {error}")
            else:
                st.session_state.ai_answers[idx] = answer

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
            st.success(f"Generate {len(st.session_state.ai_answers)} risposte AI")

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
                    st.markdown(f"**Risposta AI:**")
                    st.info(st.session_state.ai_answers[idx])

    return len(st.session_state.ai_answers) > 0


def render_section_c():
    """Sezione C: Raccolta risposte utente (ground truth)."""
    st.header("Sezione C: Risposte Ground Truth")

    if not st.session_state.ai_answers:
        st.info("Genera prima le risposte AI nella Sezione B")
        return False

    brand_name = st.session_state.brand_name
    questions = st.session_state.questions

    st.write("Inserisci le risposte corrette del brand per ogni domanda:")

    all_valid = True

    for idx in sorted(st.session_state.ai_answers.keys()):
        question = questions[idx].replace('{BRAND_NAME}', brand_name)

        # Text area per risposta utente con la domanda come label
        user_answer = st.text_area(
            question,
            value=st.session_state.user_answers.get(idx, ""),
            key=f"user_answer_{idx}",
            height=120,
            placeholder="Inserisci qui la risposta corretta del brand...",
            help="Inserisci la risposta corretta secondo il brand (min 20 caratteri)"
        )

        st.session_state.user_answers[idx] = user_answer

        # Validazione lunghezza
        if len(user_answer.strip()) < MIN_ANSWER_LENGTH:
            st.warning(f"Risposta troppo corta (min {MIN_ANSWER_LENGTH} caratteri)")
            all_valid = False

        st.divider()

    return all_valid


def render_section_d(evaluator_model: genai.GenerativeModel):
    """Sezione D: Calcolo Brand Integrity."""
    st.header("Sezione D: Calcolo Brand Integrity")

    # Verifica prerequisiti
    if not st.session_state.ai_answers:
        st.info("Genera prima le risposte AI nella Sezione B")
        return False

    if not st.session_state.user_answers:
        st.info("Inserisci le risposte ground truth nella Sezione C")
        return False

    # Verifica che tutte le risposte siano valide
    all_valid = all(
        len(st.session_state.user_answers.get(idx, "").strip()) >= MIN_ANSWER_LENGTH
        for idx in st.session_state.ai_answers.keys()
    )

    if not all_valid:
        st.warning("Completa tutte le risposte ground truth (min 20 caratteri ciascuna)")
        return False

    # Pulsante calcola
    if st.button("Calcola Brand Integrity", type="primary"):
        # Rate limiting
        can_proceed, error_msg = rate_limit_check()
        if not can_proceed:
            st.error(error_msg)
            return False

        progress_bar = st.progress(0)
        status_text = st.empty()

        st.session_state.eval_results = {}
        errors = []

        questions = st.session_state.questions
        brand_name = st.session_state.brand_name

        for idx in sorted(st.session_state.ai_answers.keys()):
            question = questions[idx].replace("{BRAND_NAME}", brand_name)
            ai_answer = st.session_state.ai_answers[idx]
            user_answer = st.session_state.user_answers[idx]

            status_text.text(f"Valutando risposta {idx + 1}/{len(st.session_state.ai_answers)}...")
            progress_bar.progress(idx / len(st.session_state.ai_answers))

            result, error = evaluate_answer(evaluator_model, question, ai_answer, user_answer)

            if error:
                errors.append(f"Domanda {idx + 1}: {error}")
            else:
                st.session_state.eval_results[idx] = result

            # Rate limiting check
            can_proceed, error_msg = rate_limit_check()
            if not can_proceed:
                st.error(error_msg)
                break

        progress_bar.progress(1.0)
        status_text.text("Valutazione completata!")

        # Calcola summary
        if st.session_state.eval_results:
            total = len(st.session_state.eval_results)
            correct = sum(1 for r in st.session_state.eval_results.values() if r['is_correct'])
            integrity_score = round((correct / total) * 100)

            st.session_state.summary = {
                'total': total,
                'correct': correct,
                'incorrect': total - correct,
                'integrity_score': integrity_score
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


def render_section_e():
    """Sezione E: Visualizzazione risultati."""
    st.header("Sezione E: Risultati")

    if not st.session_state.summary:
        st.info("Calcola prima il Brand Integrity Score nella Sezione D")
        return

    summary = st.session_state.summary

    # Score grande
    st.markdown("### Brand Integrity Score")

    # Colore basato su score
    score = summary['integrity_score']
    if score >= 80:
        color = "green"
    elif score >= 60:
        color = "orange"
    else:
        color = "red"

    st.markdown(f"<h1 style='text-align: center; color: {color};'>{score}/100</h1>", unsafe_allow_html=True)

    # Statistiche
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Totale domande", summary['total'])
    with col2:
        st.metric("Corrette", summary['correct'])
    with col3:
        st.metric("Sbagliate", summary['incorrect'])

    st.divider()

    # Tabella dettagli
    st.subheader("Dettagli per domanda")

    questions = st.session_state.questions
    brand_name = st.session_state.brand_name

    for idx in sorted(st.session_state.eval_results.keys()):
        result = st.session_state.eval_results[idx]
        question = questions[idx].replace("{BRAND_NAME}", brand_name)

        # Colore status
        if result['is_correct']:
            status = "CORRETTA"
            status_color = "green"
        else:
            status = "SBAGLIATA"
            status_color = "red"

        # Expander per ogni domanda
        with st.expander(f"Domanda {idx + 1}: {question[:60]}... - {status}"):
            st.markdown(f"**Domanda completa:** {question}")
            st.markdown(f"**Esito:** :{status_color}[{status}]")
            st.markdown(f"**Score:** {result['score']:.2f} / 1.00")
            st.markdown(f"**Motivazione:** {result['reason']}")

            if result.get('key_conflicts'):
                st.markdown("**Conflitti chiave:**")
                for conflict in result['key_conflicts']:
                    st.markdown(f"- {conflict}")

            # Mostra risposte confrontate
            with st.expander("Vedi confronto risposte"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Risposta AI:**")
                    st.info(st.session_state.ai_answers[idx])
                with col2:
                    st.markdown("**Risposta Ground Truth:**")
                    st.success(st.session_state.user_answers[idx])


def main():
    """Main app."""
    st.set_page_config(
        page_title="Brand AI Integrity Tool",
        page_icon="🎯",
        layout="wide"
    )

    st.title("Brand AI Integrity Tool")
    st.markdown("Misura la Brand Integrity confrontando risposte AI (Gemini) con risposte ground truth del brand.")

    # Init session state
    init_session_state()

    # Check secrets
    secrets_ok, error_msg = check_secrets()
    if not secrets_ok:
        st.error(f"Errore configurazione: {error_msg}")
        st.info("Configura il file .streamlit/secrets.toml seguendo l'esempio in .streamlit/secrets.toml.example")
        st.stop()

    # Configure Gemini
    gemini_model, evaluator_model, error_msg = configure_gemini()
    if error_msg:
        st.error(error_msg)
        st.stop()

    # Sidebar info
    with st.sidebar:
        st.header("Informazioni")
        st.markdown(f"""
        **Modello AI:** {st.secrets.get('GEMINI_MODEL', 'gemini-3-flash-preview')}

        **Modello Evaluator:** {st.secrets.get('EVALUATOR_MODEL', 'gemini-3-flash-preview')}

        **Soglia match:** {MATCH_THRESHOLD}

        **Chiamate API (sessione):** {st.session_state.api_calls_count}
        """)

        if st.button("Reset sessione"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Sezioni dell'app
    st.divider()

    # Sezione A: Setup
    setup_ok = render_section_a()

    st.divider()

    # Sezione B: Generazione risposte AI
    if setup_ok:
        ai_answers_ready = render_section_b(gemini_model)
    else:
        st.info("Completa la Sezione A per procedere")
        ai_answers_ready = False

    st.divider()

    # Sezione C: Risposte utente
    if ai_answers_ready:
        user_answers_ready = render_section_c()
    else:
        st.info("Genera le risposte AI per procedere")
        user_answers_ready = False

    st.divider()

    # Sezione D: Calcolo
    if user_answers_ready:
        calculation_done = render_section_d(evaluator_model)
    else:
        st.info("Completa le sezioni precedenti per calcolare il Brand Integrity Score")
        calculation_done = False

    st.divider()

    # Sezione E: Risultati
    render_section_e()

    # Footer
    st.divider()
    st.caption("Brand AI Integrity Tool - MVP v1.0")


if __name__ == "__main__":
    main()
