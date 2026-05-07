"""
Brand AI Integrity - FastAPI Backend
SSE streaming for real-time progress, PDF generation, SMTP2GO email.
"""

import os
import json
import time
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, Tuple

import asyncio

from google import genai as google_genai
from google.genai import types as genai_types
from openai import OpenAI
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import base64
import urllib.request
import urllib.parse

load_dotenv()

app = FastAPI(title="Brand AI Integrity")
app.mount("/static", StaticFiles(directory="static"), name="static")

MATCH_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.50
SOCIAL_OPTIONS = ["Instagram", "Facebook", "LinkedIn", "TikTok", "YouTube", "X (Twitter)"]

QUESTIONS = [
    {"id": "products", "label": "Indica massimo 3 prodotti/servizi principali di {BRAND_NAME}", "type": "text",
     "ai_prompt": "Quali sono i 3 principali prodotti o servizi offerti da {BRAND_NAME}? Elenca solo i 3 piu importanti."},
    {"id": "sector", "label": "In che settore opera {BRAND_NAME}?", "type": "text",
     "ai_prompt": "In quale settore opera {BRAND_NAME}?", "prefill_from": "sector"},
    {"id": "target", "label": "Qual e il pubblico target principale di {BRAND_NAME}?", "type": "text",
     "ai_prompt": "Qual e il pubblico target principale di {BRAND_NAME}?"},
    {"id": "locations", "label": "{BRAND_NAME} ha sedi operative? Se si, dove?", "type": "text",
     "ai_prompt": "{BRAND_NAME} ha sedi operative? Se si, dove si trovano?"},
    {"id": "social", "label": "Quali sono i canali social ufficiali di {BRAND_NAME}?", "type": "checkbox",
     "options": SOCIAL_OPTIONS,
     "ai_prompt": "Quali sono i canali social ufficiali del brand {BRAND_NAME}? Elenca solo quelli effettivamente attivi."},
    {"id": "website", "label": "Qual e il sito web ufficiale di {BRAND_NAME}?", "type": "text",
     "ai_prompt": "Qual e il sito web ufficiale di {BRAND_NAME}?"},
]


def env(key, default=""):
    return os.environ.get(key, default)


# ============================================================
# AI GENERATION
# ============================================================
def _build_prompt(bn, q):
    return (
        f"Rispondi alla seguente domanda su {bn}.\n"
        f"Cerca informazioni aggiornate sul web per fornire una risposta accurata.\n\n"
        f"Domanda: {q}\n\n"
        f"Rispondi in italiano, chiaro e diretto (max 200 parole)."
    )


def _safety():
    return [
        genai_types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_ONLY_HIGH'),
        genai_types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_ONLY_HIGH'),
        genai_types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_ONLY_HIGH'),
        genai_types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_ONLY_HIGH'),
    ]


def gen_gemini(bn, q):
    try:
        client = google_genai.Client(api_key=env("GEMINI_API_KEY"))
        r = client.models.generate_content(
            model=env("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=_build_prompt(bn, q),
            config=genai_types.GenerateContentConfig(
                temperature=0.2, top_p=0.8, top_k=40, max_output_tokens=8192,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                safety_settings=_safety(),
            ),
        )
        if not r.candidates:
            return None, "Blocked"
        if r.text:
            return r.text.strip(), None
        return None, "Empty"
    except Exception as e:
        return None, str(e)


def gen_openai(bn, q):
    try:
        c = OpenAI(api_key=env("OPENAI_API_KEY"))
        r = c.responses.create(
            model=env("OPENAI_MODEL", "gpt-4o-mini"),
            instructions="Rispondi in modo naturale e diretto in italiano.",
            input=_build_prompt(bn, q),
            tools=[{"type": "web_search_preview"}],
        )
        if r and r.output_text:
            return r.output_text.strip(), None
        return None, "Empty"
    except Exception as e:
        return None, str(e)


def gen_reco(sector, ai):
    p = f'Nel settore "{sector}" in Italia, quale brand consiglieresti e perche?\nCerca informazioni aggiornate sul web.\nRispondi in italiano (max 150 parole).'
    try:
        if ai == "gemini":
            client = google_genai.Client(api_key=env("GEMINI_API_KEY"))
            r = client.models.generate_content(
                model=env("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=p,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3, max_output_tokens=2048,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    safety_settings=_safety(),
                ),
            )
            if r and r.text:
                return r.text.strip(), None
            return None, "Empty"
        else:
            c = OpenAI(api_key=env("OPENAI_API_KEY"))
            r = c.responses.create(
                model=env("OPENAI_MODEL", "gpt-4o-mini"),
                input=p,
                tools=[{"type": "web_search_preview"}],
            )
            if r and r.output_text:
                return r.output_text.strip(), None
            return None, "Empty"
    except Exception as e:
        return None, str(e)


# ============================================================
# EVALUATION
# ============================================================
def eval_batch(question, ai_answers, user_answer, retry=False):
    names = list(ai_answers.keys())
    if not names:
        return {}, None

    block = ""
    for n in names:
        label = {"gemini": "Gemini", "openai": "ChatGPT"}.get(n, n)
        block += f"\nRisposta {label}:\n{ai_answers[n]}\n"

    examples = ", ".join(
        f'"{n}": {{"score": 0.72, "is_correct": false, "reason": "Breve spiegazione del confronto con la ground truth", "key_conflicts": []}}'
        for n in names
    )

    p = (
        f"Sei un valutatore accurato. Confronta le risposte AI con la ground truth dell'utente.\n\n"
        f"REGOLE FONDAMENTALI:\n"
        f"1. La ground truth contiene le informazioni che l'utente CONFERMA come corrette. "
        f"Se la risposta AI MENZIONA quegli elementi (anche con parole diverse o aggiungendo altri dettagli), e' un match.\n"
        f"2. La ground truth puo' essere abbreviata, con acronimi o parole chiave: "
        f"'seo' = SEO, 'balerna ticino' = Balerna in Canton Ticino, 'LinkedIn' = il social network LinkedIn. "
        f"Solo testo COMPLETAMENTE privo di significato (es. 'aaaaa', '12345') vale 0.0.\n"
        f"3. L'AI che aggiunge informazioni extra NON viene penalizzata. "
        f"Conta SOLO se la risposta CONTIENE gli elementi della ground truth.\n"
        f"4. PRIMA DI DARE LO SCORE: nel campo 'reason', elenca ESPLICITAMENTE quali elementi della ground truth "
        f"hai trovato PRESENTI e quali ASSENTI nella risposta AI. Poi assegna lo score.\n\n"
        f"ESEMPI DI VALUTAZIONE CORRETTA:\n"
        f"- Ground truth: 'LinkedIn' | Risposta AI: 'LinkedIn, Facebook, Instagram, YouTube' "
        f"-> score 0.90 (LinkedIn e' PRESENTE nella risposta)\n"
        f"- Ground truth: 'seo cro geo' | Risposta AI: 'SEO e Digital Advertising' "
        f"-> score 0.35 (solo SEO presente, CRO e GEO assenti = 1 su 3)\n"
        f"- Ground truth: 'balerna ticino' | Risposta AI: 'Balerna, Canton Ticino, Svizzera' "
        f"-> score 0.95 (match perfetto, stessa localita')\n"
        f"- Ground truth: 'balerna ticino' | Risposta AI: 'Milano e Londra' "
        f"-> score 0.00 (Balerna e Ticino ASSENTI)\n\n"
        f"Domanda: {question}\n{block}\n"
        f"Ground truth dell'utente: {user_answer}\n\n"
        f"Criteri di scoring:\n"
        f"- score >= 0.75: la risposta AI MENZIONA tutti o quasi tutti gli elementi della ground truth\n"
        f"- score 0.50-0.74: la risposta menziona ALCUNI elementi ma ne manca altri significativi\n"
        f"- score 0.25-0.49: la risposta menziona solo una piccola parte degli elementi\n"
        f"- score < 0.25: la risposta non menziona NESSUN elemento della ground truth\n\n"
    )
    if retry:
        p += "GENERA SOLO JSON VALIDO.\n\n"
    p += f"JSON: {{ {examples} }}"

    try:
        c = OpenAI(api_key=env("OPENAI_API_KEY"))
        r = c.chat.completions.create(
            model=env("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": p}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2048,
        )
        t = r.choices[0].message.content.strip()
        if not t:
            return None, "Empty"
        if t.startswith("```"):
            lines = t.split("\n")
            t = "\n".join(lines[1:-1]).replace("```json", "").replace("```", "").strip()
        raw = json.loads(t)
        b = {}
        for k, v in raw.items():
            kl = k.lower()
            if "openai" in kl or "chatgpt" in kl or "gpt" in kl:
                b["openai"] = v
            elif "gemini" in kl or "google" in kl:
                b["gemini"] = v
            else:
                b[k] = v
        res = {}
        for n in names:
            if n in b:
                x = b[n]
                x["score"] = float(x.get("score", 0))
                x["is_correct"] = bool(x.get("is_correct", False))
                if isinstance(x.get("reason"), str):
                    x["reason"] = x["reason"].replace("\n", " ").strip()
                x.setdefault("key_conflicts", [])
                res[n] = x

        # Sanity check: catch evaluator hallucinations with text matching
        user_elements = [e.strip().lower() for e in user_answer.replace(",", " ").split() if len(e.strip()) > 2]
        if user_elements:
            for n in list(res.keys()):
                if n not in ai_answers:
                    continue
                ai_lower = ai_answers[n].lower()
                matches = sum(1 for e in user_elements if e in ai_lower)
                match_ratio = matches / len(user_elements)
                llm_score = res[n]["score"]
                if llm_score < 0.25 and matches > 0:
                    corrected = round(match_ratio * 0.85, 2)
                    if corrected > llm_score:
                        found = [e.upper() for e in user_elements if e in ai_lower]
                        missing = [e.upper() for e in user_elements if e not in ai_lower]
                        parts = []
                        if found:
                            parts.append(f"{', '.join(found)} {'è presente' if len(found) == 1 else 'sono presenti'}")
                        if missing:
                            parts.append(f"{', '.join(missing)} {'è assente' if len(missing) == 1 else 'sono assenti'}")
                        reason = f"{matches}/{len(user_elements)} elementi trovati. {', mentre '.join(parts)}."
                        print(f"[SANITY] {n}: LLM gave {llm_score} but text match={matches}/{len(user_elements)} ({match_ratio:.0%}), correcting to {corrected}", flush=True)
                        res[n]["score"] = corrected
                        res[n]["is_correct"] = corrected >= 0.75
                        res[n]["reason"] = reason

        return res, None
    except json.JSONDecodeError:
        if not retry:
            return eval_batch(question, ai_answers, user_answer, True)
        return None, "JSON parse error"
    except Exception as e:
        return None, str(e)


def _build_comment_prompt(bn, sector, summary, eval_results):
    wrong = []
    partial_q = []
    for idx, res in eval_results.items():
        status = res.get("status", "incorrect")
        i = int(idx)
        if i < len(QUESTIONS):
            label = QUESTIONS[i]["label"].replace("{BRAND_NAME}", bn)
            if status == "incorrect":
                wrong.append(label)
            elif status == "partial":
                partial_q.append(label)

    return (
        f'Sei un esperto di brand reputation e AI.\n'
        f'Risultati Brand AI Integrity per "{bn}" (settore: {sector}):\n'
        f'- Score: {summary["integrity_score"]}/100\n'
        f'- Gemini: {summary["ai_scores"].get("gemini", 0)}/100\n'
        f'- ChatGPT: {summary["ai_scores"].get("openai", 0)}/100\n'
        f'- Errori: {", ".join(wrong) if wrong else "Nessuna"}\n'
        f'- Parziali: {", ".join(partial_q) if partial_q else "Nessuna"}\n\n'
        f'Scrivi un commento qualitativo in italiano (8-10 frasi, circa 200 parole) strutturato cosi:\n'
        f'1. Valutazione generale del risultato e cosa significa per il brand.\n'
        f'2. Quali informazioni le AI rappresentano bene e quali no (sii specifico).\n'
        f'3. Perche questo e un problema concreto per il business (es. clienti che cercano info, decisioni basate su AI).\n'
        f'4. 3-4 azioni concrete e specifiche per migliorare (es. ottimizzare schema markup, aggiornare pagina Chi Siamo, ecc.).\n'
        f'Professionale ma accessibile. No elenchi puntati, scrivi in paragrafi discorsivi.'
    )


def gen_comment_gemini(bn, sector, summary, eval_results):
    p = _build_comment_prompt(bn, sector, summary, eval_results)
    try:
        client = google_genai.Client(api_key=env("GEMINI_API_KEY"))
        r = client.models.generate_content(
            model=env("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=p,
            config=genai_types.GenerateContentConfig(
                temperature=0.3, max_output_tokens=2048,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                safety_settings=_safety(),
            ),
        )
        if r and r.text:
            return r.text.strip(), None
        return None, "Empty"
    except Exception as e:
        return None, str(e)


def gen_comment_openai(bn, sector, summary, eval_results):
    p = _build_comment_prompt(bn, sector, summary, eval_results)
    try:
        c = OpenAI(api_key=env("OPENAI_API_KEY"))
        r = c.chat.completions.create(
            model=env("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": p}],
            temperature=0.3,
            max_tokens=2048,
        )
        if r.choices and r.choices[0].message.content:
            return r.choices[0].message.content.strip(), None
        return None, "Empty"
    except Exception as e:
        return None, str(e)


# ============================================================
# ASYNC HELPERS
# ============================================================
AI_CALL_TIMEOUT = 45
EVAL_CALL_TIMEOUT = 30
MAX_RETRIES = 1
RETRY_BASE_DELAY = 3


def _is_retryable(error_str):
    if not error_str:
        return False
    e = error_str.lower()
    return any(k in e for k in ["429", "rate", "resource_exhausted", "503", "unavailable", "overloaded"])


async def _safe_call(fn, *args, timeout=AI_CALL_TIMEOUT, retries=MAX_RETRIES):
    last_error = None
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(fn, *args),
                timeout=timeout,
            )
            value, error = result
            if error and attempt < retries and _is_retryable(error):
                print(f"[RETRY] {fn.__name__} attempt {attempt + 1}: {error}", flush=True)
                await asyncio.sleep(RETRY_BASE_DELAY * (attempt + 1))
                continue
            return value, error
        except asyncio.TimeoutError:
            last_error = "Timeout"
            print(f"[TIMEOUT] {fn.__name__} attempt {attempt + 1} after {timeout}s", flush=True)
            if attempt < retries:
                await asyncio.sleep(RETRY_BASE_DELAY)
                continue
        except Exception as e:
            last_error = str(e)
            break
    return None, last_error


# ============================================================
# PDF
# ============================================================
def _cscore(s):
    if s >= 80:
        return "#4CAF50"
    if s >= 60:
        return "#FF9800"
    return "#F44336"


def _judge(s):
    if s >= 80:
        return "ECCELLENTE"
    if s >= 60:
        return "BUONO"
    return "SCARSO"


def make_pdf(bn, sector, summary, eval_results, user_answers, ai_answers, reco, comment):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=30)
    st = getSampleStyleSheet()
    ts = ParagraphStyle("T", parent=st["Heading1"], fontSize=26, textColor=colors.HexColor("#E87722"),
                        spaceAfter=10, alignment=TA_CENTER, fontName="Helvetica-Bold")
    ss = ParagraphStyle("S", parent=st["Normal"], fontSize=13, textColor=colors.HexColor("#666"),
                        spaceAfter=20, alignment=TA_CENTER)
    hs = ParagraphStyle("H", parent=st["Heading2"], fontSize=17, textColor=colors.HexColor("#E87722"),
                        spaceAfter=14, fontName="Helvetica-Bold")
    shs = ParagraphStyle("SH", parent=st["Heading3"], fontSize=13, textColor=colors.HexColor("#333"),
                         spaceAfter=10, fontName="Helvetica-Bold")
    bs = ParagraphStyle("BX", parent=st["Normal"], fontSize=9, leading=13)
    ns = st["Normal"]

    story = []
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("BRAND AI INTEGRITY REPORT", ts))
    story.append(Paragraph(f"Brand: {bn} | Settore: {sector}", ss))
    story.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y - %H:%M')}", ss))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("EXECUTIVE SUMMARY", hs))

    sc = summary["integrity_score"]
    ai_sc = summary.get("ai_scores", {})
    data = [
        ["METRICA", "VALORE", "VALUTAZIONE"],
        ["Brand AI Integrity Score", f"{sc}/100", _judge(sc)],
        ["", "", ""],
        ["Score Gemini", f"{ai_sc.get('gemini', 0)}/100", _judge(ai_sc.get("gemini", 0))],
        ["Score ChatGPT", f"{ai_sc.get('openai', 0)}/100", _judge(ai_sc.get("openai", 0))],
        ["", "", ""],
        ["Domande Totali", str(summary["total"]), ""],
        ["Corrette", str(summary["correct"]), ""],
        ["Parziali", str(summary.get("partial", 0)), ""],
        ["Da Migliorare", str(summary["incorrect"]), ""],
    ]
    t = Table(data, colWidths=[2.5 * inch, 1.5 * inch, 2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E87722")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor(_cscore(sc))),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 13),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor(_cscore(ai_sc.get("gemini", 0)))),
        ("TEXTCOLOR", (0, 3), (-1, 3), colors.white),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor(_cscore(ai_sc.get("openai", 0)))),
        ("TEXTCOLOR", (0, 4), (-1, 4), colors.white),
        ("BACKGROUND", (0, 6), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    if comment:
        story.append(Paragraph("ANALISI QUALITATIVA", hs))
        story.append(Paragraph(comment, ns))
        story.append(Spacer(1, 0.3 * inch))

    story.append(PageBreak())
    story.append(Paragraph("ANALISI DETTAGLIATA", hs))

    for idx_s in sorted(eval_results.keys(), key=lambda x: int(x)):
        idx = int(idx_s)
        result = eval_results[idx_s]
        if idx >= len(QUESTIONS):
            continue
        q = QUESTIONS[idx]
        qt = q["label"].replace("{BRAND_NAME}", bn)
        avg = result.get("average_score", 0)
        status = result.get("status", "incorrect")
        story.append(Paragraph(f"<b>DOMANDA {idx + 1}:</b> {qt}", shs))
        scl = "#4CAF50" if status == "correct" else ("#FF9800" if status == "partial" else "#F44336")
        slabel = "CORRETTA" if status == "correct" else ("PARZIALE" if status == "partial" else "DA MIGLIORARE")
        std = [["Score", f"{avg:.2f}/1.00", slabel]]
        stt = Table(std, colWidths=[1.5 * inch, 1.5 * inch, 2 * inch])
        stt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(scl)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 1, colors.white),
        ]))
        story.append(stt)
        story.append(Spacer(1, 0.1 * inch))

        ua = str(user_answers.get(str(idx), user_answers.get(idx, "N/A")))
        story.append(Paragraph("<b>Ground Truth:</b>", ns))
        gp = Paragraph(ua, bs)
        gt = Table([[gp]], colWidths=[4.5 * inch])
        gt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F5E9")),
            ("BOX", (0, 0), (-1, -1), 2, colors.HexColor("#4CAF50")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(gt)
        story.append(Spacer(1, 0.15 * inch))

        aa = ai_answers.get(str(idx), ai_answers.get(idx, {}))
        if isinstance(aa, dict):
            for an, al in [("gemini", "Gemini"), ("openai", "ChatGPT")]:
                if an in aa and an in result:
                    ar = result[an]
                    asc = ar.get("score", 0)
                    bg = "#E8F5E9" if asc >= 0.75 else ("#FFF3E0" if asc >= 0.5 else "#FFEBEE")
                    bd = "#4CAF50" if asc >= 0.75 else ("#FF9800" if asc >= 0.5 else "#F44336")
                    story.append(Paragraph(f"<b>{al}</b> (Score: {asc:.2f})", ns))
                    ap = Paragraph(aa[an], bs)
                    at = Table([[ap]], colWidths=[4.5 * inch])
                    at.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                        ("BOX", (0, 0), (-1, -1), 2, colors.HexColor(bd)),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ]))
                    story.append(at)
                    story.append(Paragraph(f"<i>{ar.get('reason', '')}</i>", ns))
                    story.append(Spacer(1, 0.1 * inch))
        story.append(Spacer(1, 0.3 * inch))

    if reco:
        story.append(PageBreak())
        story.append(Paragraph("CHI CONSIGLIANO LE AI?", hs))
        story.append(Paragraph(f'Domanda: "Nel settore {sector}, quale brand consiglieresti?"', ns))
        story.append(Spacer(1, 0.15 * inch))
        for an, al in [("gemini", "Gemini"), ("openai", "ChatGPT")]:
            if an in reco:
                story.append(Paragraph(f"<b>{al}:</b>", ns))
                story.append(Paragraph(reco[an], bs))
                story.append(Spacer(1, 0.15 * inch))

    story.append(PageBreak())
    story.append(Spacer(1, 1 * inch))
    story.append(Paragraph("Report generato da Brand AI Integrity", ns))
    story.append(Paragraph("Sviluppato dal <b>Team Innovation di AvantGrade.com</b>", ns))
    story.append(Spacer(1, 0.5 * inch))
    ctas = ParagraphStyle("CTA", parent=ns, fontSize=12, textColor=colors.white,
                          alignment=TA_CENTER, fontName="Helvetica-Bold")
    cta_link = '<link href="https://www.avantgrade.com/geo#contattaci" color="white">Vuoi migliorare? Parliamone insieme</link>'
    ct = Table([[Paragraph(cta_link, ctas)]], colWidths=[5 * inch])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E87722")),
        ("BOX", (0, 0), (-1, -1), 2, colors.HexColor("#CF6610")),
        ("TOPPADDING", (0, 0), (-1, -1), 15),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
    ]))
    story.append(ct)
    doc.build(story)
    buf.seek(0)
    return buf


# ============================================================
# EMAIL
# ============================================================
def send_email(to, bn, pdf_buf, sector="", summary=None, qualitative_comment="", nome=""):
    api_key = env("SMTP2GO_API_KEY")
    sender = env("SMTP2GO_SENDER", "noreply@avantgrade.com")
    if not api_key:
        return False, "SMTP2GO_API_KEY non configurata"

    score = summary.get("integrity_score", 0) if summary else 0
    gs = summary.get("ai_scores", {}).get("gemini", 0) if summary else 0
    cs = summary.get("ai_scores", {}).get("openai", 0) if summary else 0
    correct = summary.get("correct", 0) if summary else 0
    partial = summary.get("partial", 0) if summary else 0
    total = summary.get("total", 0) if summary else 0
    incorrect = summary.get("incorrect", 0) if summary else 0

    if score >= 80:
        score_color, score_label = "#4CAF50", "ECCELLENTE"
        score_msg = f"Ottima notizia: le AI rappresentano <b>{bn}</b> in modo chiaro e affidabile."
    elif score >= 60:
        score_color, score_label = "#FF9800", "BUONO"
        score_msg = f"Le AI conoscono <b>{bn}</b>, ma ci sono margini di miglioramento su alcune informazioni chiave."
    else:
        score_color, score_label = "#F44336", "DA MIGLIORARE"
        score_msg = f"Le AI non rappresentano correttamente <b>{bn}</b>: &egrave; il momento di intervenire."

    greeting = f"Ciao <b>{nome}</b>," if nome else "Ciao,"

    comment_html = ""
    if qualitative_comment:
        short = qualitative_comment[:300]
        if len(qualitative_comment) > 300:
            short += "..."
        comment_html = (
            f'<div style="background:#fff;border-left:4px solid #E87722;padding:16px 20px;margin:20px 0;border-radius:0 8px 8px 0;">'
            f'<p style="font-size:13px;color:#666;margin:0 0 6px;font-weight:bold;">ANALISI QUALITATIVA</p>'
            f'<p style="font-size:14px;color:#444;margin:0;line-height:1.6;">{short}</p></div>'
        )

    html_body = (
        f'<html><body style="font-family:Arial,sans-serif;color:#333;max-width:600px;margin:0 auto;background:#f4f4f4;">'
        f'<div style="background:linear-gradient(135deg,#E87722,#FF9800);padding:36px 30px;text-align:center;border-radius:10px 10px 0 0;">'
        f'<h1 style="color:white;margin:0;font-size:24px;">Brand AI Integrity Report</h1>'
        f'<p style="color:rgba(255,255,255,.9);margin:10px 0 0;font-size:16px;">{bn} &mdash; {sector}</p></div>'
        f'<div style="padding:30px;background:#f9f9f9;">'
        f'<p style="font-size:15px;line-height:1.7;color:#444;">{greeting}</p>'
        f'<p style="font-size:15px;line-height:1.7;color:#444;">'
        f'Abbiamo analizzato come <b>Gemini</b> e <b>ChatGPT</b> rappresentano <b>{bn}</b> '
        f'nel settore <b>{sector}</b>, confrontando le risposte delle AI con le informazioni reali '
        f'su <b>{total} domande chiave</b>: prodotti, target, sedi, canali social e sito web.</p>'
        f'<div style="background:{score_color};border-radius:12px;padding:24px;text-align:center;margin:24px 0;">'
        f'<p style="color:rgba(255,255,255,.85);margin:0 0 4px;font-size:13px;text-transform:uppercase;letter-spacing:1px;">Brand AI Integrity Score</p>'
        f'<p style="color:white;margin:0;font-size:48px;font-weight:bold;">{score}<span style="font-size:20px;opacity:.7">/100</span></p>'
        f'<p style="color:white;margin:6px 0 0;font-size:15px;font-weight:bold;">{score_label}</p></div>'
        f'<p style="font-size:14px;line-height:1.7;color:#444;">{score_msg}</p>'
        f'<table style="width:100%;border-collapse:collapse;margin:20px 0;">'
        f'<tr>'
        f'<td style="background:#fff;border:1px solid #eee;border-radius:8px;padding:16px;text-align:center;width:50%;">'
        f'<p style="margin:0 0 4px;font-size:12px;color:#888;">GEMINI</p>'
        f'<p style="margin:0;font-size:28px;font-weight:bold;color:{_cscore(gs)};">{gs}/100</p></td>'
        f'<td style="width:12px;"></td>'
        f'<td style="background:#fff;border:1px solid #eee;border-radius:8px;padding:16px;text-align:center;width:50%;">'
        f'<p style="margin:0 0 4px;font-size:12px;color:#888;">CHATGPT</p>'
        f'<p style="margin:0;font-size:28px;font-weight:bold;color:{_cscore(cs)};">{cs}/100</p></td>'
        f'</tr></table>'
        f'<p style="font-size:14px;color:#666;line-height:1.6;">'
        f'Su {total} domande analizzate: <b style="color:#4CAF50;">{correct} corrette</b>'
        f'{f", <b style=&quot;color:#FF9800;&quot;>{partial} parziali</b>" if partial else ""}'
        f'{f", <b style=&quot;color:#F44336;&quot;>{incorrect} da migliorare</b>" if incorrect else ""}.</p>'
        f'{comment_html}'
        f'<p style="font-size:14px;color:#666;line-height:1.6;margin-top:20px;">'
        f'In allegato trovi il <b>report PDF completo</b> con il dettaglio di ogni domanda, '
        f'le risposte di ciascuna AI e i suggerimenti per migliorare il tuo score.</p>'
        f'<hr style="border:1px solid #eee;margin:24px 0;">'
        f'<p style="text-align:center;margin:24px 0 8px;">'
        f'<a href="https://www.avantgrade.com/geo#contattaci" '
        f'style="background:#E87722;color:white;padding:14px 36px;text-decoration:none;border-radius:8px;font-weight:bold;font-size:15px;display:inline-block;">'
        f'Vuoi migliorare il tuo Score? Parliamone</a></p>'
        f'<p style="text-align:center;font-size:12px;color:#999;margin-top:20px;">'
        f'Report generato da Brand AI Integrity &mdash; Team Innovation di AvantGrade.com</p>'
        f'</div></body></html>'
    )

    try:
        pdf_buf.seek(0)
        pdf_b64 = base64.b64encode(pdf_buf.read()).decode("utf-8")

        payload = json.dumps({
            "api_key": api_key,
            "to": [to],
            "sender": sender,
            "subject": f"Brand AI Integrity Report - {bn}",
            "html_body": html_body,
            "attachments": [{
                "filename": f"Brand_AI_Integrity_{bn}.pdf",
                "fileblob": pdf_b64,
                "mimetype": "application/pdf",
            }],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.smtp2go.com/v3/email/send",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("data", {}).get("succeeded", 0) > 0:
                return True, "OK"
            return False, json.dumps(result)
    except Exception as e:
        return False, str(e)


def send_lead_notification(nome, cognome, azienda, email, bn, sector, summary, qualitative_comment, pdf_buf=None):
    api_key = env("SMTP2GO_API_KEY")
    sender = env("SMTP2GO_SENDER", "noreply@avantgrade.com")
    if not api_key:
        return False, "SMTP2GO_API_KEY non configurata"

    score = summary.get("integrity_score", 0) if summary else 0
    gs = summary.get("ai_scores", {}).get("gemini", 0) if summary else 0
    cs = summary.get("ai_scores", {}).get("openai", 0) if summary else 0
    correct = summary.get("correct", 0) if summary else 0
    partial = summary.get("partial", 0) if summary else 0
    total = summary.get("total", 0) if summary else 0
    incorrect = summary.get("incorrect", 0) if summary else 0

    if score >= 80:
        score_color, score_label = "#4CAF50", "ECCELLENTE"
    elif score >= 60:
        score_color, score_label = "#FF9800", "BUONO"
    else:
        score_color, score_label = "#F44336", "DA MIGLIORARE"

    comment_snippet = ""
    if qualitative_comment:
        short = qualitative_comment[:400]
        if len(qualitative_comment) > 400:
            short += "..."
        comment_snippet = (
            f'<div style="background:#fff;border-left:4px solid #E87722;padding:14px 18px;margin:16px 0;border-radius:0 8px 8px 0;">'
            f'<p style="font-size:12px;color:#888;margin:0 0 6px;font-weight:bold;text-transform:uppercase;">Analisi qualitativa</p>'
            f'<p style="font-size:13px;color:#444;margin:0;line-height:1.6;">{short}</p></div>'
        )

    html_body = (
        f'<html><body style="font-family:Arial,sans-serif;color:#333;max-width:600px;margin:0 auto;background:#f4f4f4;">'
        f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:30px;text-align:center;border-radius:10px 10px 0 0;">'
        f'<h1 style="color:#E87722;margin:0;font-size:22px;">Nuovo Lead - Brand AI Integrity</h1>'
        f'<p style="color:rgba(255,255,255,.7);margin:8px 0 0;font-size:14px;">Qualcuno ha appena completato l\'analisi</p></div>'
        f'<div style="padding:28px;background:#f9f9f9;">'
        f'<h3 style="color:#333;margin:0 0 16px;font-size:16px;">Dati del contatto</h3>'
        f'<table style="width:100%;border-collapse:collapse;margin-bottom:24px;">'
        f'<tr><td style="padding:10px 14px;background:#fff;border:1px solid #eee;font-weight:bold;color:#666;width:120px;">Nome</td>'
        f'<td style="padding:10px 14px;background:#fff;border:1px solid #eee;color:#333;">{nome} {cognome}</td></tr>'
        f'<tr><td style="padding:10px 14px;background:#fff;border:1px solid #eee;font-weight:bold;color:#666;">Azienda</td>'
        f'<td style="padding:10px 14px;background:#fff;border:1px solid #eee;color:#333;">{azienda}</td></tr>'
        f'<tr><td style="padding:10px 14px;background:#fff;border:1px solid #eee;font-weight:bold;color:#666;">Email</td>'
        f'<td style="padding:10px 14px;background:#fff;border:1px solid #eee;color:#333;"><a href="mailto:{email}" style="color:#E87722;">{email}</a></td></tr>'
        f'</table>'
        f'<h3 style="color:#333;margin:0 0 16px;font-size:16px;">Brand analizzato</h3>'
        f'<table style="width:100%;border-collapse:collapse;margin-bottom:24px;">'
        f'<tr><td style="padding:10px 14px;background:#fff;border:1px solid #eee;font-weight:bold;color:#666;width:120px;">Brand</td>'
        f'<td style="padding:10px 14px;background:#fff;border:1px solid #eee;color:#333;">{bn}</td></tr>'
        f'<tr><td style="padding:10px 14px;background:#fff;border:1px solid #eee;font-weight:bold;color:#666;">Settore</td>'
        f'<td style="padding:10px 14px;background:#fff;border:1px solid #eee;color:#333;">{sector}</td></tr>'
        f'</table>'
        f'<h3 style="color:#333;margin:0 0 12px;font-size:16px;">Risultato analisi</h3>'
        f'<div style="background:{score_color};border-radius:10px;padding:20px;text-align:center;margin-bottom:16px;">'
        f'<p style="color:rgba(255,255,255,.85);margin:0 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:1px;">Score</p>'
        f'<p style="color:white;margin:0;font-size:40px;font-weight:bold;">{score}<span style="font-size:18px;opacity:.7">/100</span></p>'
        f'<p style="color:white;margin:4px 0 0;font-size:14px;font-weight:bold;">{score_label}</p></div>'
        f'<table style="width:100%;border-collapse:collapse;margin-bottom:16px;">'
        f'<tr>'
        f'<td style="background:#fff;border:1px solid #eee;border-radius:8px;padding:14px;text-align:center;width:48%;">'
        f'<p style="margin:0 0 4px;font-size:11px;color:#888;">GEMINI</p>'
        f'<p style="margin:0;font-size:24px;font-weight:bold;color:{_cscore(gs)};">{gs}/100</p></td>'
        f'<td style="width:4%;"></td>'
        f'<td style="background:#fff;border:1px solid #eee;border-radius:8px;padding:14px;text-align:center;width:48%;">'
        f'<p style="margin:0 0 4px;font-size:11px;color:#888;">CHATGPT</p>'
        f'<p style="margin:0;font-size:24px;font-weight:bold;color:{_cscore(cs)};">{cs}/100</p></td>'
        f'</tr></table>'
        f'<p style="font-size:13px;color:#666;line-height:1.5;">'
        f'Su {total} domande: <b style="color:#4CAF50;">{correct} corrette</b>'
        f'{f", <b style=&quot;color:#FF9800;&quot;>{partial} parziali</b>" if partial else ""}'
        f'{f", <b style=&quot;color:#F44336;&quot;>{incorrect} da migliorare</b>" if incorrect else ""}.</p>'
        f'{comment_snippet}'
        f'<hr style="border:1px solid #eee;margin:20px 0;">'
        f'<p style="text-align:center;font-size:11px;color:#999;">'
        f'Notifica automatica &mdash; Brand AI Integrity &mdash; AvantGrade.com</p>'
        f'</div></body></html>'
    )

    try:
        email_payload = {
            "api_key": api_key,
            "to": ["brand-integrity-leads@avantgrade.com"],
            "sender": sender,
            "subject": f"Nuovo Lead: {nome} {cognome} ({azienda}) - {bn}",
            "html_body": html_body,
        }

        if pdf_buf:
            pdf_buf.seek(0)
            pdf_b64 = base64.b64encode(pdf_buf.read()).decode("utf-8")
            email_payload["attachments"] = [{
                "filename": f"Brand_AI_Integrity_{bn}.pdf",
                "fileblob": pdf_b64,
                "mimetype": "application/pdf",
            }]

        payload = json.dumps(email_payload).encode("utf-8")

        req = urllib.request.Request(
            "https://api.smtp2go.com/v3/email/send",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("data", {}).get("succeeded", 0) > 0:
                return True, "OK"
            return False, json.dumps(result)
    except Exception as e:
        return False, str(e)


# ============================================================
# ZOHO CRM
# ============================================================
def _zoho_access_token():
    cid = env("ZOHO_CLIENT_ID")
    csec = env("ZOHO_CLIENT_SECRET")
    rtok = env("ZOHO_REFRESH_TOKEN")
    if not all([cid, csec, rtok]):
        return None
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": cid,
        "client_secret": csec,
        "refresh_token": rtok,
    }).encode("utf-8")
    req = urllib.request.Request("https://accounts.zoho.com/oauth/v2/token", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8")).get("access_token")


def create_zoho_lead(nome, cognome, azienda, email, bn, sector, score):
    try:
        token = _zoho_access_token()
        if not token:
            return False, "Zoho credentials not configured"

        lead_data = {
            "data": [{
                "First_Name": nome,
                "Last_Name": cognome or "(non specificato)",
                "Company": azienda or "(non specificata)",
                "Email": email,
                "Lead_Source": "Brand Integrity",
                "Description": f"Brand analizzato: {bn}\nSettore: {sector}\nBrand AI Integrity Score: {score}/100",
            }]
        }

        payload = json.dumps(lead_data).encode("utf-8")
        req = urllib.request.Request(
            "https://www.zohoapis.com/crm/v2/Leads",
            data=payload,
            headers={
                "Authorization": f"Zoho-oauthtoken {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            status = result.get("data", [{}])[0].get("status")
            if status == "success":
                return True, "OK"
            return False, json.dumps(result)
    except Exception as e:
        return False, str(e)


# ============================================================
# ROUTES
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/questions")
async def get_questions():
    return {"questions": QUESTIONS, "social_options": SOCIAL_OPTIONS}


class AnalyzeReq(BaseModel):
    brand_name: str
    sector: str
    user_answers: dict


@app.post("/api/analyze")
async def analyze(body: AnalyzeReq):
    bn = body.brand_name
    sector = body.sector
    ua = body.user_answers

    async def stream():
        def sse(ev, d):
            return f"event: {ev}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n"

        KEEPALIVE_INTERVAL = 8

        ai_ans = {idx: {} for idx in range(len(QUESTIONS))}
        errors = []
        total = len(QUESTIONS) * 2 + len(QUESTIONS) + 3
        step = 0

        # Phase 1: Gemini + ChatGPT in parallel per question (with timeout + keepalive)
        for idx, q in enumerate(QUESTIONS):
            ai_ans[idx] = {}
            ap = q["ai_prompt"].replace("{BRAND_NAME}", bn)

            yield sse("progress", {
                "step": step, "total": total, "phase": "gemini",
                "qn": idx + 1, "qt": len(QUESTIONS),
                "msg": f"Domanda {idx + 1} di {len(QUESTIONS)}",
            })

            gather_task = asyncio.ensure_future(asyncio.gather(
                _safe_call(gen_gemini, bn, ap),
                _safe_call(gen_openai, bn, ap),
                return_exceptions=True,
            ))
            while not gather_task.done():
                done, _ = await asyncio.wait({gather_task}, timeout=KEEPALIVE_INTERVAL)
                if not done:
                    yield ": keepalive\n\n"
            results = gather_task.result()

            for ai_name, r in [("gemini", results[0]), ("openai", results[1])]:
                if isinstance(r, Exception):
                    errors.append(f"{'Gemini' if ai_name == 'gemini' else 'ChatGPT'} Q{idx + 1}: {r}")
                else:
                    a, e = r
                    if e:
                        errors.append(f"{'Gemini' if ai_name == 'gemini' else 'ChatGPT'} Q{idx + 1}: {e}")
                    elif a:
                        ai_ans[idx][ai_name] = a
            step += 2

            yield sse("progress", {
                "step": step, "total": total, "phase": "chatgpt",
                "qn": idx + 1, "qt": len(QUESTIONS),
                "msg": f"Domanda {idx + 1} di {len(QUESTIONS)}",
            })

            if idx < len(QUESTIONS) - 1:
                await asyncio.sleep(0.5)

        print(f"[ANALYSIS] Phase 1 done. AI answers: {[(k, list(v.keys())) for k, v in ai_ans.items()]}", flush=True)
        if errors:
            print(f"[ANALYSIS] Phase 1 errors: {errors}", flush=True)

        # Phase 2: Evaluate in parallel batches of 3 (with keepalive)
        ev_res = {}
        EVAL_BATCH = 3
        for batch_start in range(0, len(QUESTIONS), EVAL_BATCH):
            batch_end = min(batch_start + EVAL_BATCH, len(QUESTIONS))
            batch_indices = list(range(batch_start, batch_end))

            yield sse("progress", {
                "step": step, "total": total, "phase": "eval",
                "qn": batch_start + 1, "qt": len(QUESTIONS),
                "msg": f"Valutazione domande {batch_start + 1}-{batch_end} di {len(QUESTIONS)}",
            })

            eval_coros = []
            for idx in batch_indices:
                q = QUESTIONS[idx]
                qt = q["ai_prompt"].replace("{BRAND_NAME}", bn)
                uas = ua.get(str(idx), "")
                eval_coros.append(
                    _safe_call(eval_batch, qt, ai_ans.get(idx, {}), uas, timeout=EVAL_CALL_TIMEOUT)
                )

            eval_task = asyncio.ensure_future(asyncio.gather(*eval_coros, return_exceptions=True))
            while not eval_task.done():
                done, _ = await asyncio.wait({eval_task}, timeout=KEEPALIVE_INTERVAL)
                if not done:
                    yield ": keepalive\n\n"
            batch_results = eval_task.result()

            for idx, result in zip(batch_indices, batch_results):
                ev_res[idx] = {}
                scores = []
                if isinstance(result, Exception):
                    errors.append(f"Eval Q{idx + 1}: {result}")
                else:
                    b, be = result
                    if be:
                        errors.append(f"Eval Q{idx + 1}: {be}")
                    elif b:
                        for an in ["gemini", "openai"]:
                            if an in b:
                                ev_res[idx][an] = b[an]
                                scores.append(b[an]["score"])
                if scores:
                    avg = sum(scores) / len(scores)
                    ev_res[idx]["average_score"] = avg
                    ev_res[idx]["is_correct"] = avg >= MATCH_THRESHOLD
                    ev_res[idx]["status"] = (
                        "correct" if avg >= MATCH_THRESHOLD
                        else ("partial" if avg >= PARTIAL_THRESHOLD else "incorrect")
                    )
                print(f"[ANALYSIS] Eval Q{idx + 1}: scores={scores}", flush=True)

            step += len(batch_indices)

        # Phase 3: Recommendation - PARALLEL (with keepalive, no retry)
        yield sse("progress", {
            "step": step, "total": total, "phase": "recommendation",
            "msg": "Chi consigliano le AI?...",
        })
        reco = {}
        reco_task = asyncio.ensure_future(asyncio.gather(
            _safe_call(gen_reco, sector, "gemini", retries=0),
            _safe_call(gen_reco, sector, "openai", retries=0),
            return_exceptions=True,
        ))
        while not reco_task.done():
            done, _ = await asyncio.wait({reco_task}, timeout=KEEPALIVE_INTERVAL)
            if not done:
                yield ": keepalive\n\n"
        reco_results = reco_task.result()
        for ai_name, result in zip(["gemini", "openai"], reco_results):
            if isinstance(result, Exception):
                errors.append(f"Recommendation {ai_name}: {result}")
            else:
                a, e = result
                if not e and a:
                    reco[ai_name] = a
                elif e:
                    errors.append(f"Recommendation {ai_name}: {e}")
        step += 2

        # Phase 4: Summary
        ails = {"gemini": [], "openai": []}
        for r in ev_res.values():
            for an in ["gemini", "openai"]:
                if an in r and "score" in r[an]:
                    ails[an].append(r[an]["score"])
        aiavg = {
            an: round(sum(sc) / len(sc) * 100) if sc else 0
            for an, sc in ails.items()
        }
        integrity = round(sum(aiavg.values()) / len(aiavg)) if aiavg else 0
        correct = sum(1 for r in ev_res.values() if r.get("status") == "correct")
        partial = sum(1 for r in ev_res.values() if r.get("status") == "partial")
        incorrect = sum(1 for r in ev_res.values() if r.get("status") == "incorrect")
        summ = {
            "total": len(ev_res),
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "integrity_score": integrity,
            "ai_scores": aiavg,
        }

        # Phase 5: Qualitative comment - Gemini then OpenAI fallback (separate timeouts + keepalive)
        ev_str = {str(k): v for k, v in ev_res.items()}

        yield sse("progress", {
            "step": step, "total": total, "phase": "comment",
            "msg": "Generazione analisi qualitativa...",
        })

        comment_task = asyncio.ensure_future(
            _safe_call(gen_comment_gemini, bn, sector, summ, ev_str, timeout=25, retries=0)
        )
        while not comment_task.done():
            done, _ = await asyncio.wait({comment_task}, timeout=KEEPALIVE_INTERVAL)
            if not done:
                yield ": keepalive\n\n"
        comment, cerr = comment_task.result()

        if cerr or not comment:
            print(f"[COMMENT] Gemini failed ({cerr}), trying OpenAI...", flush=True)
            yield sse("progress", {
                "step": step, "total": total, "phase": "comment",
                "msg": "Finalizzazione analisi...",
            })
            comment_task = asyncio.ensure_future(
                _safe_call(gen_comment_openai, bn, sector, summ, ev_str, timeout=25, retries=0)
            )
            while not comment_task.done():
                done, _ = await asyncio.wait({comment_task}, timeout=KEEPALIVE_INTERVAL)
                if not done:
                    yield ": keepalive\n\n"
            comment, cerr = comment_task.result()

        if cerr or not comment:
            comment = "Analisi non disponibile."
            print(f"[COMMENT] Both AIs failed: {cerr}", flush=True)

        if errors:
            print(f"[ANALYSIS] All errors: {errors}", flush=True)
        print(f"[ANALYSIS] Final scores: integrity={integrity} gemini={aiavg.get('gemini',0)} openai={aiavg.get('openai',0)}", flush=True)

        yield sse("complete", {
            "summary": summ,
            "eval_results": {str(k): v for k, v in ev_res.items()},
            "ai_answers": {str(k): v for k, v in ai_ans.items()},
            "recommendation": reco,
            "qualitative_comment": comment,
            "errors": errors,
        })

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


class EmailReq(BaseModel):
    email: str
    nome: str = ""
    cognome: str = ""
    azienda: str = ""
    brand_name: str
    sector: str
    summary: dict
    eval_results: dict
    user_answers: dict
    ai_answers: dict
    recommendation: dict
    qualitative_comment: str


@app.post("/api/send-email")
async def email_route(b: EmailReq):
    pdf = make_pdf(
        b.brand_name, b.sector, b.summary, b.eval_results,
        b.user_answers, b.ai_answers, b.recommendation, b.qualitative_comment,
    )
    ok, msg = send_email(b.email, b.brand_name, pdf, b.sector, b.summary, b.qualitative_comment, nome=b.nome)
    print(f"[EMAIL] to={b.email} brand={b.brand_name} nome={b.nome} cognome={b.cognome} azienda={b.azienda} success={ok} msg={msg}")

    pdf.seek(0)
    lok, lmsg = send_lead_notification(
        b.nome, b.cognome, b.azienda, b.email,
        b.brand_name, b.sector, b.summary, b.qualitative_comment, pdf_buf=pdf,
    )
    print(f"[LEAD-NOTIFY] brand={b.brand_name} lead={b.nome} {b.cognome} success={lok} msg={lmsg}")

    score = b.summary.get("integrity_score", 0) if b.summary else 0
    zok, zmsg = create_zoho_lead(b.nome, b.cognome, b.azienda, b.email, b.brand_name, b.sector, score)
    print(f"[ZOHO-CRM] brand={b.brand_name} lead={b.email} success={zok} msg={zmsg}")

    return {"success": ok, "message": msg}


@app.post("/api/download-pdf")
async def pdf_route(b: EmailReq):
    pdf = make_pdf(
        b.brand_name, b.sector, b.summary, b.eval_results,
        b.user_answers, b.ai_answers, b.recommendation, b.qualitative_comment,
    )
    return Response(
        content=pdf.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Brand_AI_Integrity_{b.brand_name}.pdf"},
    )
