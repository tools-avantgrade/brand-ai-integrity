FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
ENV PORT=8080

CMD ["sh", "-c", "mkdir -p .streamlit && \
  if [ -n \"$GEMINI_API_KEY\" ]; then \
    cat > .streamlit/secrets.toml <<EOF2\nGEMINI_API_KEY = \"$GEMINI_API_KEY\"\nGEMINI_MODEL = \"${GEMINI_MODEL:-gemini-1.5-flash}\"\nEVALUATOR_MODEL = \"${EVALUATOR_MODEL:-gemini-1.5-flash}\"\nEOF2\n  fi && \
  streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
