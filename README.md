# Voice Assistant with LiveKit, LlamaIndex and Ollama

A practice voice assistant that answers questions from your own documents. LlamaIndex indexes the files in `docs/` and persists the index to `chat-engine-storage/`, a local Ollama model generates answers, and LiveKit handles the voice session.

## Setup

```bash
pip install -r requirements.txt
```n
Install [Ollama](https://ollama.com) and pull a model, add your documents to a `docs/` folder, put your LiveKit credentials in `.env`, then run `python app.py`.

## Tech stack

Python, LiveKit, LlamaIndex, Ollama, Hugging Face embeddings
