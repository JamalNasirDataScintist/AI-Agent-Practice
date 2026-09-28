"""
Voice Assistant with LiveKit + LlamaIndex + Ollama
Python 3.10 compatible
"""

# =====================================================
# 1. Environment & Logging
# =====================================================
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-assistant")

PERSIST_DIR = "./chat-engine-storage"
DOCS_DIR = "./docs"

# =====================================================
# 2. LlamaIndex Setup
# =====================================================
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.ollama import Ollama

def build_index():
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    llm = Ollama(
        model="gemma",
        request_timeout=120.0,
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    if not os.path.exists(PERSIST_DIR):
        if not os.path.exists(DOCS_DIR):
            raise RuntimeError("docs/ folder is missing")

        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        if not documents:
            raise RuntimeError("docs/ folder is empty")

        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR
        )
        index = load_index_from_storage(storage_context)

    return index.as_chat_engine(chat_mode=ChatMode.CONTEXT)

chat_engine = build_index()

# =====================================================
# 3. LiveKit Prewarm (VAD)
# =====================================================
from livekit.agents import JobProcess
from livekit.plugins import silero

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Silero VAD loaded")

# =====================================================
# 4. RAG Wrapper (CRITICAL FIX)
# =====================================================
async def rag_llm(prompt: str) -> str:
    """
    Wrap LlamaIndex ChatEngine so LiveKit can use it
    """
    response = chat_engine.chat(prompt)
    return response.response

# =====================================================
# 5. Voice Assistant Entrypoint
# =====================================================
from livekit.agents import (
    JobContext,
    WorkerOptions,
    AutoSubscribe,
)
from livekit.agents
