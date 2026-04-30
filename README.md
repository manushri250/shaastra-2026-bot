# Shaastra Bot 2026

A Streamlit dashboard chatbot for answering questions about Shaastra 2026 events, workshops, competitions, schedules, venues, and speakers from the provided context document.

## Local Setup

Clone the repository:

```bash
git clone https://github.com/manushri250/Shaastra_Bot_2026_v1.git
cd Shaastra_Bot_2026_v1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the model folder:

```bash
mkdir models
```

Download the local LLM model and place it at:

```text
models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

Model download link:

```text
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

Run the dashboard:

```bash
python -m streamlit run dashboard.py
```

Open the local dashboard URL:

```text
http://localhost:8501
```

## Notes

- The app uses `shaastra_contextdoc.pdf` as the source document.
- The embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
- The local LLM is `mistral-7b-instruct-v0.2.Q4_K_M.gguf`.
- The `models/` folder is intentionally not included in GitHub because the GGUF model is several GB.
- If the model file is missing, the app still runs with fallback retrieved-context answers, but full local LLM answers require the GGUF model.
