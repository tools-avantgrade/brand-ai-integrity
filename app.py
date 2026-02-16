"""
Brand AI Integrity Tool - MVP

Misura la Brand Integrity del brand confrontando risposte AI (Gemini, ChatGPT, Claude)
con risposte ground truth fornite dall'utente.
"""

import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime


# Configurazione
MATCH_THRESHOLD = 0.75
DEFAULT_QUESTIONS = [
    "Quali sono i principali prodotti/servizi offerti da {BRAND_NAME}?",
    "In che settore opera {BRAND_NAME}?",
    "Qual è il pubblico target principale di {BRAND_NAME}?",
    "{BRAND_NAME} ha sedi operative? Dove?",
    "Quali sono i canali social ufficiali del brand {BRAND_NAME}? (ad esempio linkedin, instagram)",
    "Quali sono i contatti del brand {BRAND_NAME}? (Ad es. inserisci il numero di telefono, pagine contatto del sito)"
]
MIN_QUESTIONS = 3
MAX_QUESTIONS = 10
# MIN_ANSWER_LENGTH = 20  # DEPRECATED: Rimossa validazione lunghezza minima


def init_session_state():
    """Inizializza lo stato della sessione Streamlit."""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1  # Step: 1=Brand, 2=Domande, 3=Risultati
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


def get_all_questions():
    """Restituisce tutte le domande (default + personalizzate)."""
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

        # Verifica Brave Search API Key (per web search - GRATIS, no limiti)
        brave_key = st.secrets["BRAVE_API_KEY"]
        if not brave_key or brave_key == "YOUR_BRAVE_API_KEY_HERE":
            return False, "BRAVE_API_KEY non configurata correttamente in secrets.toml"

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

        # Config per modello di generazione - aumentato token limit
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,  # Aumentato per evitare risposte troncate
        }

        # Config per evaluator: più token e response JSON mode
        evaluator_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }

        # Safety settings per Gemini (più permissivi per evitare blocchi)
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        gemini_model = genai.GenerativeModel(
            model_name=gemini_model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
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


@st.cache_data(ttl=300, show_spinner=False)  # Cache 5 minuti per risultati più freschi
def web_search(query: str, max_results: int = 10) -> Tuple[str, bool]:
    """
    Effettua ricerca web usando Brave Search API (GRATIS, usata da molte AI).

    Brave Search è l'API usata da ChatGPT e altre AI per accesso web reale.

    Args:
        query: Query di ricerca
        max_results: Numero massimo di risultati (default 10)

    Returns:
        Tupla (risultati formattati, successo)
    """
    try:
        brave_key = st.secrets.get("BRAVE_API_KEY", "")
        if not brave_key:
            return "Brave API key non configurata.", False

        # Brave Search API endpoint
        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": brave_key
        }

        params = {
            "q": query,
            "count": max_results,
            "search_lang": "it",  # Priorità risultati italiani
            "text_decorations": False,
            "safesearch": "moderate"
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Estrai risultati web
        web_results = data.get('web', {}).get('results', [])

        if not web_results:
            return "Nessun risultato trovato dalla ricerca web.", False

        # Formatta i risultati in modo naturale (come ChatGPT li vede)
        formatted_results = ""

        for idx, result in enumerate(web_results[:max_results], 1):
            title = result.get('title', 'N/A')
            description = result.get('description', 'N/A')
            url_link = result.get('url', '')

            formatted_results += f"{idx}. {title}\n"
            formatted_results += f"{description}\n"
            formatted_results += f"Fonte: {url_link}\n\n"

        return formatted_results.strip(), True

    except requests.exceptions.RequestException as e:
        return f"Errore connessione Brave Search: {str(e)}", False
    except Exception as e:
        return f"Errore durante la ricerca web: {str(e)}", False


def generate_pdf_report(brand_name: str, summary: Dict, eval_results: Dict, questions: List[str], user_answers: Dict, ai_answers: Dict) -> BytesIO:
    """
    Genera un report PDF completo e professionale del Brand AI Integrity Score.

    Returns:
        BytesIO: Buffer contenente il PDF generato
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=30)

    # Styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=10,
        alignment=1,
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#666666'),
        spaceAfter=20,
        alignment=1
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=15,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )

    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#333333'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )

    # Stile per testo nelle box colorate - semplificato per word wrap affidabile
    box_text_style = ParagraphStyle(
        'BoxText',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        leftIndent=0,
        rightIndent=0,
        spaceBefore=0,
        spaceAfter=0,
        alignment=TA_LEFT
    )

    # Build PDF content
    story = []

    # === TITLE PAGE ===
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("BRAND AI INTEGRITY REPORT", title_style))
    story.append(Paragraph(f"Brand: {brand_name}", subtitle_style))
    story.append(Paragraph(f"Data Analisi: {datetime.now().strftime('%d/%m/%Y - %H:%M')}", subtitle_style))
    story.append(Spacer(1, 0.5*inch))

    # === EXECUTIVE SUMMARY ===
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))

    score = summary['integrity_score']
    ai_scores = summary.get('ai_scores', {})

    # Tabella score principale
    summary_data = [
        ['METRICA', 'VALORE', 'VALUTAZIONE'],
        ['Punteggio complessivo di Brand AI Integrity', f"{score}/100", get_judgment(score)],
        ['', '', ''],
        ['Score Gemini', f"{ai_scores.get('gemini', 0)}/100", get_judgment(ai_scores.get('gemini', 0))],
        ['Score ChatGPT', f"{ai_scores.get('openai', 0)}/100", get_judgment(ai_scores.get('openai', 0))],
        ['Score Claude', f"{ai_scores.get('claude', 0)}/100", get_judgment(ai_scores.get('claude', 0))],
        ['', '', ''],
        ['Domande Totali', str(summary['total']), ''],
        ['Risposte Corrette (media)', str(summary['correct']), ''],
        ['Risposte da Migliorare (media)', str(summary['incorrect']), ''],
    ]

    t = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
    t.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),

        # Score principale - colorato
        ('BACKGROUND', (0, 1), (-1, 1), get_color_for_score(score)),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.white),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 14),

        # Score AI individuali
        ('BACKGROUND', (0, 3), (-1, 3), get_color_for_score(ai_scores.get('gemini', 0))),
        ('TEXTCOLOR', (0, 3), (-1, 3), colors.white),
        ('BACKGROUND', (0, 4), (-1, 4), get_color_for_score(ai_scores.get('openai', 0))),
        ('TEXTCOLOR', (0, 4), (-1, 4), colors.white),
        ('BACKGROUND', (0, 5), (-1, 5), get_color_for_score(ai_scores.get('claude', 0))),
        ('TEXTCOLOR', (0, 5), (-1, 5), colors.white),

        # Resto tabella
        ('BACKGROUND', (0, 7), (-1, -1), colors.beige),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))

    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # === LEGENDA ===
    story.append(Paragraph("<b>Legenda Valutazioni:</b>", styles['Normal']))
    legend_data = [
        ['', 'ECCELLENTE (80-100)', 'BUONO (60-79)', 'SCARSO (<60)'],
    ]
    legend_table = Table(legend_data, colWidths=[0.5*inch, 1.8*inch, 1.8*inch, 1.8*inch])
    legend_table.setStyle(TableStyle([
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#4CAF50')),
        ('BACKGROUND', (2, 0), (2, 0), colors.HexColor('#FF9800')),
        ('BACKGROUND', (3, 0), (3, 0), colors.HexColor('#F44336')),
        ('TEXTCOLOR', (1, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(legend_table)

    # === DETTAGLI PER DOMANDA ===
    story.append(PageBreak())
    story.append(Paragraph("ANALISI DETTAGLIATA PER DOMANDA", heading_style))
    story.append(Paragraph("Confronto tra risposte Ground Truth (del brand) e risposte delle AI", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    for idx in sorted(eval_results.keys()):
        result = eval_results[idx]
        question = questions[idx].replace("{BRAND_NAME}", brand_name)
        avg_score = result.get('average_score', 0)
        is_correct = result.get('is_correct', False)

        # === HEADER DOMANDA ===
        story.append(Paragraph(f"<b>DOMANDA {idx + 1}</b>", subheading_style))
        story.append(Paragraph(question, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

        # Score medio domanda
        score_status = "✓ CORRETTA" if is_correct else "✗ DA MIGLIORARE"
        score_color_hex = '#4CAF50' if is_correct else '#F44336'

        status_data = [['Punteggio complessivo', f"{avg_score:.2f}/1.00", score_status]]
        status_table = Table(status_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        status_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(score_color_hex)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ]))
        story.append(status_table)
        story.append(Spacer(1, 0.15*inch))

        # === GROUND TRUTH ===
        story.append(Paragraph("<b>✓ RISPOSTA GROUND TRUTH (Brand):</b>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

        # Usa Paragraph per word wrap automatico con larghezza ridotta e sicura
        gt_text = Paragraph(user_answers[idx], box_text_style)
        ground_truth_data = [[gt_text]]
        gt_table = Table(ground_truth_data, colWidths=[4.5*inch])
        gt_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E8F5E9')),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#4CAF50')),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(gt_table)
        story.append(Spacer(1, 0.2*inch))

        # === RISPOSTE AI ===
        story.append(Paragraph("<b>RISPOSTE DELLE AI:</b>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

        if idx in ai_answers:
            ai_ans = ai_answers[idx]

            for ai_name, ai_label, ai_icon in [
                ("gemini", "Gemini", "⚫"),
                ("openai", "ChatGPT", "🟢"),
                ("claude", "Claude", "🟣")
            ]:
                if ai_name in ai_ans and ai_name in result:
                    ai_result = result[ai_name]
                    ai_score = ai_result.get('score', 0)
                    ai_reason = ai_result.get('reason', 'N/A')
                    ai_is_correct = ai_result.get('is_correct', False)

                    # Determina colore
                    if ai_score >= 0.75:
                        bg_color = colors.HexColor('#E8F5E9')  # Verde chiaro
                        border_color = colors.HexColor('#4CAF50')
                        status_text = "✓ CORRETTA"
                    elif ai_score >= 0.5:
                        bg_color = colors.HexColor('#FFF3E0')  # Arancione chiaro
                        border_color = colors.HexColor('#FF9800')
                        status_text = "⚠ PARZIALE"
                    else:
                        bg_color = colors.HexColor('#FFEBEE')  # Rosso chiaro
                        border_color = colors.HexColor('#F44336')
                        status_text = "✗ SBAGLIATA"

                    # Header AI
                    story.append(Paragraph(f"<b>{ai_icon} {ai_label}</b> - Score: {ai_score:.2f}/1.00 - {status_text}", styles['Normal']))
                    story.append(Spacer(1, 0.08*inch))

                    # Risposta AI - usa Paragraph per word wrap automatico - larghezza ridotta e sicura
                    ai_answer_para = Paragraph(ai_ans[ai_name], box_text_style)
                    ai_data = [[ai_answer_para]]
                    ai_table = Table(ai_data, colWidths=[4.5*inch])
                    ai_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                        ('BOX', (0, 0), (-1, -1), 2, border_color),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                        ('TOPPADDING', (0, 0), (-1, -1), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ]))
                    story.append(ai_table)
                    story.append(Spacer(1, 0.08*inch))

                    # Motivazione valutazione
                    story.append(Paragraph(f"<i>Motivazione: {ai_reason}</i>", styles['Normal']))

                    # Gap analysis se parziale o sbagliata
                    if ai_score < 0.75:
                        gap_text = f"<b>⚠ GAP IDENTIFICATO:</b> Differenza con ground truth: {(1 - ai_score) * 100:.0f}%"
                        story.append(Paragraph(gap_text, styles['Normal']))

                    story.append(Spacer(1, 0.15*inch))

        story.append(Spacer(1, 0.25*inch))

        # Separatore tra domande
        story.append(Paragraph("_" * 100, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

    # === FOOTER ===
    story.append(PageBreak())
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("Report generato da Brand AI Integrity Tool", styles['Normal']))
    story.append(Paragraph("Sviluppato dal <b>Team Innovation di AvantGrade.com</b>", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Data generazione: {datetime.now().strftime('%d/%m/%Y - %H:%M:%S')}", styles['Normal']))

    # Call to Action - Contatta AvantGrade
    story.append(Spacer(1, 0.5*inch))

    # Stile per il bottone CTA
    cta_style = ParagraphStyle(
        'CTA',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.white,
        alignment=1,  # Center
        fontName='Helvetica-Bold'
    )

    cta_text = '<link href="mailto:info@avantgrade.com?subject=Interesse%20Brand%20AI%20Integrity&amp;body=Ciao%2C%0A%0Aho%20usato%20il%20Brand%20AI%20Integrity%20Tool%20e%20vorrei%20saperne%20di%20pi%C3%B9%20su%20come%20migliorare%20la%20presenza%20del%20mio%20brand%20nelle%20AI.%0A%0AGrazie!" color="white">📧 Vuoi migliorare la tua Brand AI Integrity? Contatta AvantGrade</link>'
    cta_para = Paragraph(cta_text, cta_style)

    # Crea una table per il bottone arancione
    cta_data = [[cta_para]]
    cta_table = Table(cta_data, colWidths=[5*inch])
    cta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FF9800')),  # Arancione
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#F57C00')),  # Bordo arancione scuro
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    story.append(cta_table)
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph('<i>Clicca sul bottone per inviarci una email a info@avantgrade.com</i>', styles['Normal']))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def get_color_for_score(score: int) -> colors.Color:
    """Restituisce il colore appropriato per uno score."""
    if score >= 80:
        return colors.HexColor('#4CAF50')  # Verde
    elif score >= 60:
        return colors.HexColor('#FF9800')  # Arancione
    else:
        return colors.HexColor('#F44336')  # Rosso


def get_judgment(score: int) -> str:
    """Restituisce il giudizio per uno score."""
    if score >= 80:
        return "ECCELLENTE"
    elif score >= 60:
        return "BUONO"
    else:
        return "SCARSO"


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

        # Controlla se la risposta è stata bloccata dai safety filters
        if not response.candidates:
            return None, "Risposta bloccata dai safety filters di Gemini"

        candidate = response.candidates[0]

        # Controlla il finish_reason
        finish_reason = candidate.finish_reason
        if finish_reason and str(finish_reason) != "FinishReason.STOP":
            # La risposta è stata troncata o bloccata
            if "SAFETY" in str(finish_reason):
                return None, f"Risposta bloccata per motivi di sicurezza: {finish_reason}"
            elif "MAX_TOKENS" in str(finish_reason):
                return None, "Risposta troncata: limite token raggiunto"

        # Estrai il testo
        if response.text:
            return response.text.strip(), None
        else:
            return None, "Risposta vuota da Gemini"

    except Exception as e:
        return None, f"Errore Gemini: {str(e)}"


@st.cache_data(ttl=300, show_spinner=False)  # Cache ridotta per freschi risultati
def generate_openai_answer(_client: OpenAI, brand_name: str, question: str) -> Tuple[Optional[str], Optional[str]]:
    """Genera risposta da ChatGPT con ricerca web REALE (Brave Search)."""
    try:
        final_question = question.replace("{BRAND_NAME}", brand_name)
        openai_model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

        # Ricerca web con Brave (stessa API usata da ChatGPT)
        search_query = f"{brand_name} {final_question}"
        search_results, search_success = web_search(search_query, max_results=10)

        if search_success:
            # Simula esattamente come ChatGPT usa la web search
            system_prompt = """Sei ChatGPT con accesso alla ricerca web. Rispondi basandoti principalmente sulle informazioni attuali trovate sul web.
Comportati esattamente come faresti nell'app ChatGPT quando hai accesso alla ricerca web."""

            user_prompt = f"""Ho cercato sul web per rispondere alla tua domanda.

Domanda: {final_question}

Ecco cosa ho trovato sul web:
{search_results}

Rispondi in italiano in modo naturale e completo (massimo 200 parole), usando le informazioni trovate.
Non dire "basandomi sui risultati" - rispondi in modo diretto come se tu avessi trovato queste informazioni."""

        else:
            # Fallback alla conoscenza interna
            system_prompt = "Sei ChatGPT. Rispondi basandoti sulla tua conoscenza."
            user_prompt = f"""{final_question}

Rispondi in italiano (massimo 200 parole). Non ho trovato risultati recenti sul web, quindi usa la tua conoscenza generale."""

        response = _client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )

        if response and response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip(), None
        else:
            return None, "Risposta vuota da ChatGPT"
    except Exception as e:
        return None, f"Errore ChatGPT: {str(e)}"


@st.cache_data(ttl=300, show_spinner=False)  # Cache ridotta per freschi risultati
def generate_claude_answer(_client: Anthropic, brand_name: str, question: str) -> Tuple[Optional[str], Optional[str]]:
    """Genera risposta da Claude con ricerca web REALE (Brave Search)."""
    try:
        final_question = question.replace("{BRAND_NAME}", brand_name)
        claude_model = st.secrets.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

        # Ricerca web con Brave (stessa qualità delle app AI)
        search_query = f"{brand_name} {final_question}"
        search_results, search_success = web_search(search_query, max_results=10)

        if search_success:
            # Simula esattamente come Claude userebbe la web search
            user_prompt = f"""Ciao! Ho bisogno di informazioni su questa domanda: {final_question}

Ho effettuato una ricerca web e ho trovato queste informazioni:

{search_results}

Per favore, rispondi in italiano (massimo 200 parole) basandoti su queste informazioni trovate sul web.
Rispondi in modo naturale e diretto, come se tu avessi accesso web integrato."""

        else:
            # Fallback alla conoscenza interna
            user_prompt = f"""{final_question}

Non sono riuscito a trovare informazioni recenti sul web per "{brand_name}".
Rispondi basandoti sulla tua conoscenza in italiano (massimo 200 parole), specificando che le informazioni potrebbero non essere le più recenti."""

        response = _client.messages.create(
            model=claude_model,
            max_tokens=1024,
            temperature=0.3,
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


def render_step_1_brand():
    """Step 1: Inserimento brand name."""
    st.subheader("Step 1: Inserisci il nome del tuo Brand")

    brand_name = st.text_input(
        "Nome del Brand",
        value=st.session_state.brand_name,
        placeholder="es. Nike, Apple, AvantGrade...",
        help="Inserisci il nome del brand da analizzare",
        key="brand_input"
    )

    if brand_name and brand_name != st.session_state.brand_name:
        st.session_state.brand_name = brand_name
        # Reset quando cambia brand
        st.session_state.ai_answers = {}
        st.session_state.user_answers = {}
        st.session_state.eval_results = {}
        st.session_state.summary = None

    if brand_name:
        st.success(f"✓ Brand selezionato: **{brand_name}**")

        if st.button("Continua →", type="primary"):
            st.session_state.current_step = 2
            st.rerun()
    else:
        st.info("Inserisci il nome del brand per continuare")


def render_step_2_questions_answers(gemini_model, openai_client, anthropic_client, evaluator_model):
    """Step 2: Domande, risposte utente, generazione AI e calcolo (tutto in uno)."""
    brand_name = st.session_state.brand_name

    st.subheader(f"Step 2: Rispondi alle domande su {brand_name}")
    st.markdown("Fornisci le risposte corrette secondo il tuo brand. Le AI cercheranno queste informazioni sul web.")

    questions = get_all_questions()

    # Usa le domande default (non più modificabili)
    base_questions = DEFAULT_QUESTIONS

    # Sezione domande e risposte
    all_valid = True

    for idx, q in enumerate(base_questions):
        question = q.replace("{BRAND_NAME}", brand_name)

        with st.container():
            st.markdown(f"**Domanda {idx + 1}**")

            # Mostra la domanda (disabled)
            st.text_area(
                "",
                value=question,
                key=f"default_question_{idx}",
                height=60,
                disabled=True,
                label_visibility="collapsed"
            )

            user_answer = st.text_area(
                "La tua risposta",
                value=st.session_state.user_answers.get(idx, ""),
                key=f"user_answer_{idx}",
                height=100,
                placeholder="Inserisci la risposta corretta secondo il tuo brand..."
            )

            st.session_state.user_answers[idx] = user_answer

            # Validazione: solo risposta vuota
            if len(user_answer.strip()) == 0:
                all_valid = False

            st.markdown("---")

    # Check if all answers are valid
    if all_valid:
        st.success("✓ Tutte le risposte sono complete!")

        # Bottone per generare e calcolare tutto insieme
        if st.button("🚀 Analizza con le AI e Calcola Brand Integrity", type="primary"):
            with st.spinner("🔄 Analisi in corso..."):
                # Stima tempo: ~6 secondi per domanda x 3 AI + 3 secondi valutazione
                estimated_time = len(questions) * 20  # secondi stimati

                # Container per il timer
                timer_container = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                start_time = time.time()

                # Mostra stima iniziale
                timer_container.markdown(
                    f"<div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; text-align: center; margin: 15px 0; border: 3px solid #1976D2;'>"
                    f"<h2 style='margin: 0; color: #1976D2;'>⏱️ Analisi in corso...</h2>"
                    f"<p style='margin: 10px 0; font-size: 1.3em; font-weight: bold; color: #1976D2;'>Tempo stimato: ~{estimated_time} secondi</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.session_state.ai_answers = {}
                errors = []

                total_steps = len(questions) * 3
                current_step_count = 0

                for idx, question in enumerate(questions):
                    st.session_state.ai_answers[idx] = {}

                    # Gemini
                    elapsed = int(time.time() - start_time)
                    remaining = max(0, estimated_time - elapsed)
                    timer_container.markdown(
                        f"<div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; text-align: center; margin: 15px 0; border: 3px solid #1976D2;'>"
                        f"<h2 style='margin: 0; color: #1976D2;'>⚫ Gemini - Domanda {idx + 1}/{len(questions)}</h2>"
                        f"<p style='margin: 10px 0; font-size: 1.3em; font-weight: bold; color: #1976D2;'>⏱️ Tempo trascorso: {elapsed}s | Tempo stimato rimanente: ~{remaining}s</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    status_text.text(f"⚫ Gemini: elaborazione domanda {idx + 1} di {len(questions)}...")
                    progress_bar.progress(current_step_count / total_steps / 2)

                    gemini_answer, gemini_error = generate_gemini_answer(gemini_model, brand_name, question)
                    if gemini_error:
                        errors.append(f"Gemini Q{idx + 1}: {gemini_error}")
                    else:
                        st.session_state.ai_answers[idx]["gemini"] = gemini_answer
                    current_step_count += 1

                    # ChatGPT
                    elapsed = int(time.time() - start_time)
                    remaining = max(0, estimated_time - elapsed)
                    timer_container.markdown(
                        f"<div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center; margin: 15px 0; border: 3px solid #2E7D32;'>"
                        f"<h2 style='margin: 0; color: #2E7D32;'>🟢 ChatGPT - Domanda {idx + 1}/{len(questions)}</h2>"
                        f"<p style='margin: 10px 0; font-size: 1.3em; font-weight: bold; color: #2E7D32;'>⏱️ Tempo trascorso: {elapsed}s | Tempo stimato rimanente: ~{remaining}s</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    status_text.text(f"🟢 ChatGPT: elaborazione domanda {idx + 1} di {len(questions)}...")
                    progress_bar.progress(current_step_count / total_steps / 2)

                    openai_answer, openai_error = generate_openai_answer(openai_client, brand_name, question)
                    if openai_error:
                        errors.append(f"ChatGPT Q{idx + 1}: {openai_error}")
                    else:
                        st.session_state.ai_answers[idx]["openai"] = openai_answer
                    current_step_count += 1

                    # Claude
                    elapsed = int(time.time() - start_time)
                    remaining = max(0, estimated_time - elapsed)
                    timer_container.markdown(
                        f"<div style='background-color: #F3E5F5; padding: 20px; border-radius: 10px; text-align: center; margin: 15px 0; border: 3px solid #7B1FA2;'>"
                        f"<h2 style='margin: 0; color: #7B1FA2;'>🟣 Claude - Domanda {idx + 1}/{len(questions)}</h2>"
                        f"<p style='margin: 10px 0; font-size: 1.3em; font-weight: bold; color: #7B1FA2;'>⏱️ Tempo trascorso: {elapsed}s | Tempo stimato rimanente: ~{remaining}s</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    status_text.text(f"🟣 Claude: elaborazione domanda {idx + 1} di {len(questions)}...")
                    progress_bar.progress(current_step_count / total_steps / 2)

                    claude_answer, claude_error = generate_claude_answer(anthropic_client, brand_name, question)
                    if claude_error:
                        errors.append(f"Claude Q{idx + 1}: {claude_error}")
                    else:
                        st.session_state.ai_answers[idx]["claude"] = claude_answer
                    current_step_count += 1

                # Step 2: Valuta risposte
                elapsed = int(time.time() - start_time)
                remaining = max(0, estimated_time - elapsed)
                timer_container.markdown(
                    f"<div style='background-color: #FFF3E0; padding: 20px; border-radius: 10px; text-align: center; margin: 15px 0; border: 3px solid #E65100;'>"
                    f"<h2 style='margin: 0; color: #E65100;'>📊 Valutazione risposte in corso...</h2>"
                    f"<p style='margin: 10px 0; font-size: 1.3em; font-weight: bold; color: #E65100;'>⏱️ Tempo trascorso: {elapsed}s | Quasi finito!</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                status_text.text("Valutando le risposte con AI evaluator...")
                st.session_state.eval_results = {}

                ai_models = ["gemini", "openai", "claude"]
                total_evals = len(st.session_state.ai_answers) * len(ai_models)
                current_eval = 0

                for idx in sorted(st.session_state.ai_answers.keys()):
                    question = questions[idx].replace("{BRAND_NAME}", brand_name)
                    ai_answers = st.session_state.ai_answers[idx]
                    user_answer = st.session_state.user_answers[idx]

                    st.session_state.eval_results[idx] = {}
                    scores = []

                    for ai_name in ai_models:
                        if ai_name in ai_answers:
                            progress_bar.progress(0.5 + (current_eval / total_evals / 2))

                            result, error = evaluate_answer(evaluator_model, question, ai_answers[ai_name], user_answer)

                            if error:
                                errors.append(f"Eval Q{idx + 1} ({ai_name}): {error}")
                            else:
                                st.session_state.eval_results[idx][ai_name] = result
                                scores.append(result['score'])

                            current_eval += 1

                    # Average score
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        st.session_state.eval_results[idx]['average_score'] = avg_score
                        st.session_state.eval_results[idx]['is_correct'] = avg_score >= MATCH_THRESHOLD

                # Step 3: Calcola summary
                if st.session_state.eval_results:
                    total = len(st.session_state.eval_results)

                    # Calcola score per ogni AI
                    ai_scores = {ai: [] for ai in ai_models}
                    for result in st.session_state.eval_results.values():
                        for ai_name in ai_models:
                            if ai_name in result and 'score' in result[ai_name]:
                                ai_scores[ai_name].append(result[ai_name]['score'])

                    ai_averages = {
                        ai: round(sum(scores) / len(scores) * 100) if scores else 0
                        for ai, scores in ai_scores.items()
                    }

                    # Score medio CORRETTO: media degli score delle 3 AI
                    gemini_avg = ai_averages.get('gemini', 0)
                    openai_avg = ai_averages.get('openai', 0)
                    claude_avg = ai_averages.get('claude', 0)
                    integrity_score = round((gemini_avg + openai_avg + claude_avg) / 3)

                    # Conta risposte corrette (per statistiche)
                    correct = sum(1 for r in st.session_state.eval_results.values() if r.get('is_correct', False))

                    st.session_state.summary = {
                        'total': total,
                        'correct': correct,
                        'incorrect': total - correct,
                        'integrity_score': integrity_score,
                        'ai_scores': ai_averages
                    }

                progress_bar.progress(1.0)

                # Calcola tempo totale
                total_time = int(time.time() - start_time)

                # Mostra messaggio finale con tempo
                timer_container.markdown(
                    f"<div style='background-color: #C8E6C9; padding: 25px; border-radius: 10px; text-align: center; margin: 15px 0; border: 3px solid #2E7D32;'>"
                    f"<h1 style='margin: 0; color: #2E7D32;'>✅ Analisi completata!</h1>"
                    f"<p style='margin: 15px 0; font-size: 1.4em; font-weight: bold; color: #2E7D32;'>⏱️ Tempo totale: {total_time} secondi</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                status_text.empty()
                progress_bar.empty()

                if errors:
                    st.warning("⚠️ Alcuni errori durante l'elaborazione:")
                    for err in errors[:5]:  # Mostra solo i primi 5
                        st.text(err)

                time.sleep(2)  # Mostra il messaggio di successo per 2 secondi

                # Passa allo step 3
                st.session_state.current_step = 3
                st.rerun()
    else:
        st.info("Completa tutte le risposte per procedere con l'analisi")


def render_step_3_results():
    """Step 3: Visualizzazione risultati con PDF export."""
    if not st.session_state.summary:
        st.error("Nessun risultato disponibile")
        if st.button("← Torna indietro"):
            st.session_state.current_step = 2
            st.rerun()
        return

    brand_name = st.session_state.brand_name
    summary = st.session_state.summary

    st.subheader(f"📊 Risultati: Brand AI Integrity per {brand_name}")

    # AI Logos e modelli (mostrati solo qui nei risultati)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**⚫ Gemini**")
        st.caption(f"Modello: {st.secrets.get('GEMINI_MODEL', 'gemini-3-flash-preview')}")
    with col2:
        st.markdown("**🟢 ChatGPT**")
        st.caption(f"Modello: {st.secrets.get('OPENAI_MODEL', 'gpt-4o-mini')}")
    with col3:
        st.markdown("**🟣 Claude**")
        st.caption(f"Modello: {st.secrets.get('CLAUDE_MODEL', 'claude-sonnet-4-5-20250929')}")

    st.markdown("---")

    # Score principale con box colorato (MEDIO delle 3 AI)
    score = summary['integrity_score']
    if score >= 80:
        color = "#4CAF50"  # Verde
        emoji = "🟢"
        judgment = "ECCELLENTE"
        message = "Ottimo lavoro: l'AI rappresenta il brand in modo chiaro, coerente e affidabile. 😎"
    elif score >= 60:
        color = "#FF9800"  # Arancione
        emoji = "🟡"
        judgment = "BUONO"
        message = "Buono, ma puoi fare di meglio! Il brand è generalmente rappresentato in modo corretto, ma sono presenti alcune imprecisioni o incoerenze migliorabili."
    else:
        color = "#F44336"  # Rosso
        emoji = "🔴"
        judgment = "SCARSO"
        message = "Non ci siamo! 😭 Le risposte dell'AI risultano spesso inaccurate o incoerenti e non rappresentano correttamente il brand. Che ne dici di fare due chiacchiere?"

    st.markdown(
        f"""
        <div style='background-color: {color}; padding: 30px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white; margin: 0; font-size: 1.2em; opacity: 0.9;'>Punteggio complessivo di Brand AI Integrity</h3>
            <h1 style='color: white; margin: 10px 0; font-size: 4em;'>{emoji} {score}/100</h1>
            <h2 style='color: white; margin: 0;'>{judgment}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mostra messaggio descrittivo sotto il box
    st.markdown(
        f"<p style='text-align: center; font-size: 1.2em; margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 8px; color: #333333;'>{message}</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Score per AI - Responsive (si impilano su schermi piccoli)
    st.markdown("### 📈 Performance per AI")

    ai_scores = summary.get('ai_scores', {})

    # Usa CSS responsive per le card
    col1, col2, col3 = st.columns(3)

    with col1:
        gemini_score = ai_scores.get('gemini', 0)
        g_color = "#4CAF50" if gemini_score >= 80 else ("#FF9800" if gemini_score >= 60 else "#F44336")
        st.markdown(
            f"""
            <div style='background-color: {g_color}; padding: 20px; border-radius: 10px; text-align: center; min-height: 120px;'>
                <h3 style='color: white; margin: 0; font-size: 1.1em;'>⚫ Gemini</h3>
                <h1 style='color: white; margin: 10px 0 0 0; font-size: 2.5em;'>{gemini_score}/100</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        openai_score = ai_scores.get('openai', 0)
        o_color = "#4CAF50" if openai_score >= 80 else ("#FF9800" if openai_score >= 60 else "#F44336")
        st.markdown(
            f"""
            <div style='background-color: {o_color}; padding: 20px; border-radius: 10px; text-align: center; min-height: 120px;'>
                <h3 style='color: white; margin: 0; font-size: 1.1em;'>🟢 ChatGPT</h3>
                <h1 style='color: white; margin: 10px 0 0 0; font-size: 2.5em;'>{openai_score}/100</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        claude_score = ai_scores.get('claude', 0)
        c_color = "#4CAF50" if claude_score >= 80 else ("#FF9800" if claude_score >= 60 else "#F44336")
        st.markdown(
            f"""
            <div style='background-color: {c_color}; padding: 20px; border-radius: 10px; text-align: center; min-height: 120px;'>
                <h3 style='color: white; margin: 0; font-size: 1.1em;'>🟣 Claude</h3>
                <h1 style='color: white; margin: 10px 0 0 0; font-size: 2.5em;'>{claude_score}/100</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Domande totali", summary['total'])
    with col2:
        st.metric("✅ Risposte corrette (media)", summary['correct'])
    with col3:
        st.metric("❌ Risposte sbagliate (media)", summary['incorrect'])

    st.markdown("---")

    # Dettagli per domanda (expander)
    st.markdown("### 🔍 Dettagli per Domanda")

    questions = get_all_questions()

    for idx in sorted(st.session_state.eval_results.keys()):
        result = st.session_state.eval_results[idx]
        question = questions[idx].replace("{BRAND_NAME}", brand_name)

        avg_score = result.get('average_score', 0)
        status = "✅ CORRETTA" if result.get('is_correct', False) else "❌ SBAGLIATA"

        with st.expander(f"Domanda {idx + 1}: {question[:50]}... - {status}"):
            st.markdown(f"**Domanda:** {question}")
            st.markdown(f"**Score medio:** {avg_score:.2f} / 1.00")

            st.markdown("---")

            # Ground truth
            st.markdown("**✅ La tua risposta (Ground Truth):**")
            st.success(st.session_state.user_answers[idx])

            st.markdown("---")

            # Risposte AI - layout responsive (una sotto l'altra per leggibilità)
            if idx in st.session_state.ai_answers:
                ai_ans = st.session_state.ai_answers[idx]

                # Gemini
                st.markdown("**⚫ Gemini:**")
                if "gemini" in ai_ans:
                    st.info(ai_ans["gemini"])
                    if "gemini" in result:
                        g_res = result["gemini"]
                        st.markdown(f"✓ Score: {g_res.get('score', 0):.2f} - {g_res.get('reason', '')}")
                else:
                    st.warning("Non disponibile")

                st.markdown("")  # Spacing

                # ChatGPT
                st.markdown("**🟢 ChatGPT:**")
                if "openai" in ai_ans:
                    st.info(ai_ans["openai"])
                    if "openai" in result:
                        o_res = result["openai"]
                        st.markdown(f"✓ Score: {o_res.get('score', 0):.2f} - {o_res.get('reason', '')}")
                else:
                    st.warning("Non disponibile")

                st.markdown("")  # Spacing

                # Claude
                st.markdown("**🟣 Claude:**")
                if "claude" in ai_ans:
                    st.info(ai_ans["claude"])
                    if "claude" in result:
                        c_res = result["claude"]
                        st.markdown(f"✓ Score: {c_res.get('score', 0):.2f} - {c_res.get('reason', '')}")
                else:
                    st.warning("Non disponibile")

    # === CALL TO ACTION - Contatta AvantGrade ===
    st.markdown("---")
    st.markdown("### 📧 Vuoi migliorare la tua Brand AI Integrity?")

    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin: 20px 0;'>
            <h2 style='color: white; margin: 0 0 15px 0; font-size: 1.8em;'>
                🚀 Migliora la presenza del tuo brand nelle AI
            </h2>
            <p style='color: white; font-size: 1.2em; margin: 0 0 20px 0; opacity: 0.95;'>
                Il Team Innovation di AvantGrade può aiutarti a ottimizzare la rappresentazione del tuo brand
                nelle intelligenze artificiali e migliorare il tuo Brand AI Integrity Score.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mailto_url = (
            "mailto:info@avantgrade.com"
            "?subject=Interesse%20Brand%20AI%20Integrity"
            "&body=Ciao%2C%0A%0A"
            "ho%20usato%20il%20Brand%20AI%20Integrity%20Tool%20e%20vorrei%20saperne%20di%20pi%C3%B9%20su%20come%20migliorare%20la%20presenza%20del%20mio%20brand%20nelle%20AI.%0A%0A"
            "Grazie!"
        )
        st.markdown(
            f"""
            <a href="{mailto_url}" target="_blank" style="
                display: block;
                text-align: center;
                background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
                color: white;
                padding: 15px 30px;
                border-radius: 10px;
                text-decoration: none;
                font-size: 1.2em;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.15);
                transition: transform 0.2s, box-shadow 0.2s;
            " onmouseover="this.style.transform='scale(1.03)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.2)';"
               onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.15)';">
                📧 Contatta AvantGrade.com
            </a>
            """,
            unsafe_allow_html=True
        )


def main():
    """Main app con flusso step-by-step."""
    st.set_page_config(
        page_title="Brand AI Integrity Tool",
        page_icon="🎯",
        layout="centered"  # Cambiato da "wide" a "centered"
    )

    # Init session state
    init_session_state()

    # Header
    st.title("🎯 Brand AI Integrity Tool")
    st.markdown("**Misura quanto le risposte dell'AI rappresentano correttamente il tuo brand.**")
    st.markdown("---")

    # Check secrets
    secrets_ok, error_msg = check_secrets()
    if not secrets_ok:
        st.error(f"⚠️ Errore configurazione: {error_msg}")
        st.info("Configura le chiavi API in secrets.toml: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, BRAVE_API_KEY")
        st.stop()

    # Configure AI models
    gemini_model, openai_client, anthropic_client, evaluator_model, error_msg = configure_ai_models()
    if error_msg:
        st.error(error_msg)
        st.stop()

    # Progress indicator (senza emoji nei numeri)
    current_step = st.session_state.current_step
    steps = ["Brand", "Domande & Risposte", "Risultati"]
    st.progress((current_step - 1) / 2)
    st.markdown(f"**Passo {current_step}/3:** {steps[current_step - 1]}")
    st.markdown("---")

    # STEP 1: Brand Name
    if current_step == 1:
        render_step_1_brand()

    # STEP 2: Domande e Risposte
    elif current_step == 2:
        render_step_2_questions_answers(gemini_model, openai_client, anthropic_client, evaluator_model)

    # STEP 3: Risultati
    elif current_step == 3:
        render_step_3_results()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
        "Sviluppato dal <b>Team Innovation di AvantGrade.com</b>"
        "</div>",
        unsafe_allow_html=True
    )

    # Reset button (piccolo, in basso)
    if st.button("🔄 Ricomincia", help="Resetta l'analisi e ricomincia da capo"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
