import json
import os
import requests
from dotenv import load_dotenv
import streamlit as st
from sseclient import SSEClient

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat", page_icon="🤖", layout="centered")
st.title("RAG Chat (Bedrock)")

# sidebar controls
use_rag = st.sidebar.checkbox("Use RAG (retrieval)", value=True)
top_k = st.sidebar.slider("Top-K passages", min_value=1, max_value=6, value=3, step=1)
st.sidebar.caption(f"Backend: {BACKEND_URL}")

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# chat input
user_input = st.chat_input("Ask me anything…")

if user_input:
    # display the user message and save it
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # assistant area
    with st.chat_message("assistant"):
        text_box = st.empty()
        accumulated = ""

        # building short-term memory
        raw_hist = st.session_state.messages[:-1]

        def clip(txt: str, max_chars: int = 1200) -> str:
            return txt if len(txt) <= max_chars else (txt[:max_chars] + " …")

        history_payload = [
            {"role": role, "content": clip(content)}
            for role, content in raw_hist[-6:]  # last 6 turns
        ]

        # request payload
        payload = {
            "message": user_input,
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.2,
            "max_tokens": 512,
            "use_rag": use_rag,
            "top_k": top_k,
            "history": history_payload,
        }

        stream_url = f"{BACKEND_URL}/chat/stream"

        # stream with a spinner loading icon
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    stream_url,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    stream=True,
                    timeout=60,
                )
                messages = SSEClient(resp)
                for event in messages.events():
                    if not event.data:
                        continue
                    data = json.loads(event.data)
                    if "token" in data:
                        accumulated += data["token"]
                        text_box.markdown(accumulated)
                    if data.get("event") == "done":
                        break
            except Exception as e:
                text_box.markdown(f"Streaming error: {e}")

    st.session_state.messages.append(("assistant", accumulated))
