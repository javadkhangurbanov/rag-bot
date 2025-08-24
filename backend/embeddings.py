import os
import json
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(retries={"max_attempts": 3, "mode": "standard"}),
)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Titan v2 text embeddings via Bedrock.
    Request body: {"inputText": "string"}
    Response body: {"embedding": [floats...]}
    """
    vectors: list[list[float]] = []
    for t in texts:
        body = {"inputText": t}
        resp = _bedrock.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        out = json.loads(resp["body"].read())
        vectors.append(out["embedding"])
    return vectors
