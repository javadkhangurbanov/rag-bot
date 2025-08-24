FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
