# VoiceOps 2026 — ZentroMart Support Agent

A four-stage AI voice-support prototype built for the VoiceOps 2026 capstone.
Ollama runs the language model and embedding model locally; Gnani.ai provides
all speech-to-text and text-to-speech. The project uses only the capstone's
synthetic records and mock tools.

## Repository

Public GitHub URL: **[https://github.com/RituSheelvant/voiceops-2026-zentromart?utm_source=chatgpt.com]**

---

## Architecture

```
Customer audio → Gnani STT → SupportAgent (RAG + tools + conversation state)
             → Gnani TTS → saved audio reply
```

| Stage | Implementation |
|-------|---------------|
| Step 1 — LLM baseline | `answer_prompt()` with no retriever (prompt-only) |
| Step 2 — RAG | `Retriever` indexes 21 KB docs + products.csv via `nomic-embed-text`; `resolve_from_evidence()` applies policy precedence before the LLM is called |
| Step 3 — Tools & conversation | `SupportAgent` manages caller verification, order state, ticket creation, and multi-turn history |
| Step 4 — Gnani speech | `run_audio_batch()` / `web_ui.py` use `gnani-vachana` SDK, `Kaveri` voice, `timbre-v2.5` model, configurable language |

---

## Prerequisites

- Python 3.9 or later
- [Ollama](https://ollama.com) installed and running
- The extracted capstone dataset in the expected location (see Dataset setup)
- `GNANI_API_KEY` set as a Windows user environment variable — **never in source code**

---

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
python run.py --check
python .\Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026\validate_dataset.py
```

If the dataset folder is elsewhere, pass `--dataset <path>` (the folder
directly containing `README.md` and `validate_dataset.py`).

---

## Environment variables

Copy `.env.example` to `.env` (or set them as Windows user variables):

| Variable | Default | Purpose |
|----------|---------|---------|
| `GNANI_API_KEY` | *(required)* | Gnani speech services |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Chat model name |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

---

## Running the agent

```powershell
# Answer one typed question
python run.py --text "When can I return a pair of shoes?"

# Multi-turn text chat
python run.py --chat

# Compare baseline vs RAG on 52 practice questions
python run.py --evaluate

# Run 23 practice conversations (96 turns)
python run.py --eval-conversations

# Limit to N rows for a quick check
python run.py --evaluate --limit 3
```

---

## Required batch-run command

Place customer audio files (WAV / MP3 / M4A / FLAC / OGG) in a folder, then:

```powershell
python run.py --audio-folder recordings --output sample_outputs --language en-IN
```

For every audio file the runner saves beneath `sample_outputs/<clip-name>/`:
- `transcript.txt` — Gnani STT output
- `reply.txt` — agent reply text
- `reply.wav` — Gnani TTS audio (Kaveri / timbre-v2.5)
- `sources.json` — retrieved section IDs, decision, and tool calls
- `error.txt` — error message (only if processing failed)

A `batch_summary.json` is written in the output folder with the status of
every file. A failed file is clearly reported; it does not abort remaining
files.

Gnani STT uses the selected `--language`. TTS always uses `timbre-v2.5` and
the `Kaveri` voice. Each recording must be within 60 seconds.

---

## Spoken conversation interface

```powershell
python run.py --web --model qwen2.5:3b --language en-IN
```

Binds only to `127.0.0.1`. Allow microphone access, press **Start speaking**,
then **Stop and send**. Displays transcript, retrieved sources, and plays the
spoken reply. Press **New conversation** to reset state.

---

## Dataset and safety rules

- Reference date: 20 October 2026. Business days are Monday–Saturday,
  excluding the listed holidays; return windows use calendar days.
- Policy precedence: `pol_returns_v4.md` (current) overrides
  `pol_returns_v3_superseded.md` (superseded). Support logs are not
  authoritative policy.
- Order details are revealed only after the caller provides the last four
  digits of the registered phone number **or** the delivery pincode. One retry
  is allowed; verification proofs are never logged in conversation history.
- A ticket handover is confirmed only when `create_ticket()` returns
  `created=True`.
- Gnani only. No alternative speech service is used.
- Synthetic data only. No real customer records or recordings.

---

## Evaluation results

### Historical pre-fix full Qwen 3B evaluation (frozen in `sample_outputs/question_evaluation.json` and `sample_outputs/conversation_evaluation.json`)

| Mode | Decision match | All-required-facts | Evidence hit |
|------|---------------|-------------------|-------------|
| Prompt-only (Step 1) | 9 / 52 | 8 / 52 | 0 / 47 (no retrieval) |
| RAG (Step 2) | 33 / 52 | 37 / 52 | 47 / 47 |
| Conversations (Step 3) | 96 / 96 | 96 / 96 | — |

Tool checks: 96/96. Forbidden-content hits: 0.

### Post-fix saved-context replay (NOT a fresh model run — see `question_postfix_replay.json` and `conversation_postfix_replay.json`)

After `resolve_from_evidence()` policy patches, replays applied new rules to
prior RAG retrievals. **These are not fresh Ollama inferences.**

| Mode | Decision match | All-required-facts | Evidence hit |
|------|---------------|-------------------|-------------|
| RAG replay | 52 / 52 | 52 / 52 | 47 / 47 |
| Conversations replay | 96 / 96 | 96 / 96 | — |

### Fresh post-fix evaluation

A fresh full-model evaluation was started via `python run.py --evaluate`.
Results will be saved in `sample_outputs/question_evaluation.json` (the file
will be overwritten when the run completes). If Ollama inference is too slow
to finish within the available time, the partial results are reported
transparently; the saved-context replay scores above are **not** presented as
fresh results.

---

## Known limitations

1. **Evaluation speed.** Qwen 3B on CPU takes several minutes per inference
   call. A full 52-question × 2-mode evaluation requires Ollama to be running
   and can take several hours.

2. **Replay vs. fresh.** Post-fix replay scores reuse prior model retrievals
   where the patch was applied. They are clearly distinguished in all outputs.

---


## Dataset validator

```
PASS — all 9 CSVs, 21 KB documents, primary/foreign keys, order math,
       practice evidence, verification gates and company naming validated.
       1,000 orders | 1,783 items | 244 logs | 52 questions | 23 conversations
```
