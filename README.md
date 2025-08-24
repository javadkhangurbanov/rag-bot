# RAG Chatbot (FastAPI + Streamlit + AWS Bedrock)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot, built during bootcamp exercises.  
It is split into two services:
- **Backend**: FastAPI application that connects to Amazon Bedrock LLMs, performs retrieval from a local knowledge base, and streams responses.
- **Frontend**: Streamlit UI to interact with the chatbot in real time.

The project is fully containerized with Docker and deployed on an AWS EC2 instance.  
CI/CD is handled via GitHub Actions with linting, Docker builds, and auto-deploy to EC2.

---


---

## Features
- **Chat with LLMs via Amazon Bedrock** (Claude models by default).
- **Optional RAG pipeline**: Titan Embeddings + ChromaDB retrieval.
- **Streaming responses** from backend to frontend.
- **Conversation memory**.
- **Dockerized** backend and frontend with Compose orchestration.
- **Deployed on EC2** with public access.
- **CI/CD pipeline**:
  - Runs code quality checks (`ruff`, `black`, `isort`).
  - Builds backend/frontend Docker images.
  - Deploys automatically to EC2 via self-hosted runner.

---

## Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/rag-bot.git
   cd rag-bot
   ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Linux/Mac
    .venv\Scripts\activate      # Windows
    ```

3. Create a `.env` file in the project root:
    ```bash
    AWS_REGION=us-east-1
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
    EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
    CHROMA_DIR=./data/chroma
    ```

4. Install dependencies and run backend:
    ```bash
    pip install -r requirements.txt
    uvicorn backend.main:app --reload --port 8000
    ```

5. In another terminal, run frontend:
    ```bash
    cd frontend
    export BACKEND_URL=http://localhost:8000
    streamlit run app.py
    ```

---

## Running with Docker

Build and run both services:
    ```bash
    docker compose up --build
    ```

Backend(health): http://localhost:8000/health
Frontend: http://localhost:8501

---

## Deployment on AWS EC2

1. Clone the repo on your EC2 instance.
2. Create `.env` file with your AWS/Bedrock configuration.
3. Run:
    ```bash
    docker compose up --build -d
    ```
4. Access via:
    - Backend(health): `http://<EC2_PUBLIC_IP>:8000/health` | For my EC2 instance the public ip is: `54.87.174.113`
    - Frontend: `http://<EC2_PUBLIC_IP>:8501`

---

## CI/CD with GitHub Actions

- On pull requests -> lints code and builds Docker images.
On push to `main` -> deploys automatically to EC2 using a self-hosted runner.

---

## Assets

Screenshots of the running frontend and backend(health) can be found under:

- `assets/frontend`
- `assets/backend`

---

## Example

- Add `.md` files into `knowledge/`
- Run:
    ```bash
    python -m backend.ingest
    ```
- Ask the bot questions -> if context matches, it retrieves and cites the file.