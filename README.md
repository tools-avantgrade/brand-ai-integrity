# Brand AI Integrity Tool

Tool Streamlit per misurare la **Brand Integrity** del tuo brand sui motori AI (Gemini).

## Cos'è

Il tool pone una serie di domande sul tuo brand a Gemini AI e confronta le risposte con le risposte ground truth fornite da te. Calcola poi un **Brand Integrity Score** (0-100) basato sulla coerenza tra le due fonti.

## Caratteristiche

- Interfaccia semplice e intuitiva con Streamlit
- Domande personalizzabili (3-10 domande)
- Confronto semantico robusto tramite AI evaluator
- Caching delle risposte per evitare chiamate duplicate
- Rate limiting per rispettare i limiti API
- Risultati dettagliati con motivazioni e conflitti

## Requisiti

- Python 3.8 o superiore
- Account Google Cloud con Gemini API abilitata
- API Key Gemini (gratuita su [Google AI Studio](https://aistudio.google.com/app/apikey))

## Setup Locale

### 1. Clona la repository

```bash
git clone <repository-url>
cd brand-ai-integrity
```

### 2. Crea ambiente virtuale (opzionale ma consigliato)

```bash
python -m venv venv
source venv/bin/activate  # Su Linux/Mac
# oppure
venv\Scripts\activate  # Su Windows
```

### 3. Installa dipendenze

```bash
pip install -r requirements.txt
```

### 4. Configura i secrets

Copia il file di esempio e inserisci la tua API key:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Modifica `.streamlit/secrets.toml` con un editor di testo:

```toml
GEMINI_API_KEY = "la-tua-api-key-qui"
GEMINI_MODEL = "gemini-1.5-flash"  # Opzionale
EVALUATOR_MODEL = "gemini-1.5-flash"  # Opzionale
```

**IMPORTANTE:** Il file `.streamlit/secrets.toml` è git-ignored per sicurezza. Non committare mai questo file!

### 5. Avvia l'applicazione

```bash
streamlit run app.py
```

L'app si aprirà automaticamente nel browser su `http://localhost:8501`

## Setup su Streamlit Cloud

### 1. Deploy su Streamlit Cloud

1. Fai push della repository su GitHub
2. Vai su [share.streamlit.io](https://share.streamlit.io)
3. Collega la tua repository
4. Seleziona `app.py` come main file

### 2. Configura i secrets

Nel dashboard di Streamlit Cloud:

1. Vai su **Settings** > **Secrets**
2. Inserisci i secrets nel formato TOML:

```toml
GEMINI_API_KEY = "la-tua-api-key-qui"
GEMINI_MODEL = "gemini-1.5-flash"
EVALUATOR_MODEL = "gemini-1.5-flash"
```

3. Salva e redeploy

## Come usare il tool

### Sezione A: Setup

1. Inserisci il **nome del brand** da analizzare
2. Personalizza le **domande** (default: 5 domande):
   - Usa `{BRAND_NAME}` come placeholder per il nome del brand
   - Aggiungi o rimuovi domande (min 3, max 10)

### Sezione B: Risposte AI

1. Clicca **"Genera risposte AI"**
2. Il tool interroga Gemini per ogni domanda
3. Le risposte AI vengono salvate e mostrate

### Sezione C: Risposte Ground Truth

1. Per ogni domanda, inserisci la **risposta corretta** secondo il brand
2. Minimo 20 caratteri per risposta
3. Queste sono le tue "ground truth" per il confronto

### Sezione D: Calcolo Brand Integrity

1. Clicca **"Calcola Brand Integrity"**
2. L'AI evaluator confronta ogni coppia di risposte
3. Assegna uno score (0-1) e determina se è corretta (soglia: 0.75)

### Sezione E: Risultati

- **Brand Integrity Score**: punteggio finale (0-100)
- **Dettagli per domanda**: esito, score, motivazioni, conflitti
- **Confronto risposte**: vedi AI vs Ground Truth affiancate

## Configurazione Modelli

### Modelli disponibili (Google Gemini)

- `gemini-1.5-flash` (default): veloce ed economico
- `gemini-1.5-pro`: più capace, più lento e costoso
- `gemini-2.0-flash-exp`: sperimentale, performance migliorate

### Parametri di generazione

Il tool usa questi parametri per risposte stabili:

- **Temperature**: 0.2 (bassa variabilità)
- **Top-p**: 0.8
- **Top-k**: 40
- **Max tokens**: 1024

## Costi e Rate Limiting

### Costi API Gemini

- **Gemini 1.5 Flash**: ~$0.075 per 1M token input, $0.30 per 1M token output
- **Free tier**: 15 richieste/minuto, 1500 richieste/giorno, 1M token/giorno

Per una sessione tipica (5 domande):
- ~10 chiamate API totali
- ~5000-10000 token totali
- **Costo stimato**: $0.001-0.005 (praticamente gratis nel free tier)

Fonte: [Google AI Pricing](https://ai.google.dev/pricing)

### Rate Limiting

Il tool implementa:

- **Limite sessione**: max 30 chiamate/minuto
- **Caching**: risposte AI cachate per 1 ora (evita duplicati)
- **Retry logic**: gestione errori con retry automatico per l'evaluator

## Architettura

### File Structure

```
brand-ai-integrity/
├── app.py                           # Applicazione Streamlit completa
├── requirements.txt                 # Dipendenze Python
├── README.md                        # Questa documentazione
├── .gitignore                       # Ignora secrets e cache
└── .streamlit/
    ├── secrets.toml.example         # Template secrets
    └── secrets.toml                 # Secrets reali (git-ignored)
```

### Logica di Evaluation

L'evaluator Gemini riceve:
- Domanda originale
- Risposta AI
- Risposta ground truth (utente)

Restituisce JSON:
```json
{
  "score": 0.85,
  "is_correct": true,
  "reason": "La risposta AI è semanticamente allineata...",
  "key_conflicts": []
}
```

**Criterio di correttezza**: `score >= 0.75`

### Session State

Lo stato della sessione Streamlit include:
- `brand_name`: nome del brand
- `questions`: lista domande
- `ai_answers`: dict {idx: risposta_ai}
- `user_answers`: dict {idx: risposta_utente}
- `eval_results`: dict {idx: evaluation_result}
- `summary`: summary finale (totale, corrette, score)

## Troubleshooting

### Errore: "GEMINI_API_KEY mancante"

Verifica che il file `.streamlit/secrets.toml` esista e contenga la chiave:
```toml
GEMINI_API_KEY = "your-key-here"
```

### Errore: "Rate limit reached"

Aspetta 60 secondi o riduci il numero di domande. Il limite è 30 chiamate/minuto per sessione.

### Errore: "Parsing JSON failed"

L'evaluator a volte restituisce testo non-JSON. Il tool fa automaticamente 1 retry. Se persiste, verifica la stabilità del modello o prova un modello più capace.

### Risposta AI troppo generica

- Usa domande più specifiche
- Fornisci ground truth dettagliate
- Prova un modello più capace (gemini-1.5-pro)

## Limitazioni

- **Lingua**: ottimizzato per italiano, ma supporta altre lingue
- **Soggettività**: l'evaluator AI può avere bias
- **Costi**: usa API a pagamento (ma free tier generoso)
- **Rate limits**: max 30 chiamate/min, 1500/giorno (free tier)
- **Context**: risposte AI limitate a knowledge cutoff del modello

## Roadmap Futuri Miglioramenti

- [ ] Support per più AI providers (OpenAI, Claude, Mistral)
- [ ] Export risultati in PDF/CSV
- [ ] Storico valutazioni nel tempo
- [ ] Comparazione multi-brand
- [ ] Dashboard analytics avanzato
- [ ] Integrazione con Google Search per fact-checking

## Licenza

MIT License - vedi LICENSE file

## Autori

Sviluppato con Claude Code

## Support

Per bug, feature request o domande, apri una issue su GitHub.

---

**Versione:** 1.0 (MVP)
**Data:** 2026-01-20
