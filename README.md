# VoiceOps 2026 — ZentroMart Support Agent

A local prototype for the VoiceOps capstone. Ollama runs the language model and
embedding model locally. Gnani is used for speech-to-text and text-to-speech.
The project uses only the capstone's synthetic records and mock tools.

## Repository

Public GitHub URL: add your repository URL after publishing the repository.

## Prerequisites

- Python 3.9 or later.
- Ollama installed and running.
- The extracted capstone dataset beside this project folder. Do not copy the
  dataset into the submission ZIP.
- A personal Gnani API key saved as the Windows user environment variable
  GNANI_API_KEY. Never put the real key in source control.

## Setup (Windows PowerShell)

From the project folder:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    ollama pull qwen2.5:3b
    ollama pull nomic-embed-text
    python run.py --check
    python .\Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026\validate_dataset.py

If the dataset folder is elsewhere, provide its direct folder path with
--dataset. It is the folder containing README.md and validate_dataset.py.

## Run the agent

Ask one text question:

    python run.py --text "When can I return a pair of shoes?"

Start a multi-turn text conversation:

    python run.py --chat

Compare prompt-only answers with retrieval-augmented answers on the practice
questions:

    python run.py --evaluate

Run the multi-turn practice conversations:

    python run.py --eval-conversations

The quick check can limit the number of cases, for example:
python run.py --evaluate --limit 3.

## Spoken conversation interface

Start the local microphone-and-playback page:

    python run.py --web --model qwen2.5:3b --language en-IN

Allow microphone access, press Start speaking, then Stop and send. The interface
uses Gnani STT and Gnani TTS, displays the transcript and retrieved sources,
plays the spoken reply, and keeps conversation context for follow-up turns.
Each recording is limited to 60 seconds. Press New conversation to clear state.
The page binds only to 127.0.0.1.

## Required batch run

Put customer audio clips in a folder, then run:

    python run.py --audio-folder recordings --output sample_outputs --language en-IN

Supported input extensions are WAV, MP3, M4A, FLAC and OGG. For every clip,
the agent saves the transcript, reply text, retrieved section IDs, and reply
audio beneath sample_outputs/<clip-name>/. Gnani STT uses the selected
language; TTS explicitly uses timbre-v2.5 and voice Kaveri.

## Dataset and safety rules

- The dataset reference date is 20 October 2026. Business days are Monday
  through Saturday, excluding the listed holidays; return windows use
  calendar days.
- Knowledge sections retain their IDs and document metadata. Current policy
  takes precedence over agent guides, help, FAQs and marketing. The
  superseded returns policy applies only to the cases described by its
  successor.
- An order lookup returns no private order facts until a phone suffix or
  delivery pincode is verified. The agent allows one retry. It does not read
  verification values or private contact details back to the customer.
- The mock tools are for synthetic training data only, not production
  authentication. A ticket handover is confirmed only when the tool returns
  created=True.
- The speech layer uses Gnani only. No real customer data or recordings are
  used.

## Evaluation and verification status

The dataset validator passes: 9 CSVs and 21 KB documents, including 1,000 orders, 1,783 order items, 244 logs, 52 practice questions and 23 conversations. `python run.py --check` confirms Ollama is reachable and `GNANI_API_KEY` is set.

A full qwen2.5:3b model evaluation was completed before the latest answer-rule patch. It scored prompt-only at 9/52 accepted decisions and 8/52 all-required-facts; RAG at 33/52 accepted decisions, 37/52 all-required-facts and 47/47 gold-evidence hits. The conversation suite scored 96/96 facts, 96/96 decisions, 96/96 expected-tool checks and zero forbidden-content hits. These are the pre-patch baseline results in `question_evaluation.json` and `conversation_evaluation.json`.

After the patch, a saved-context replay applied the new evidence-based policy rules to the prior RAG retrievals and affected conversation turns. It scored 52/52 RAG required-fact and accepted-decision checks, with 47/47 applicable gold-evidence hits; the conversation replay scored 96/96 facts, 96/96 decisions, 96/96 tool checks and zero forbidden-content hits. These are incremental replays, not fresh full model runs. See `question_postfix_replay.json` and `conversation_postfix_replay.json`.

The updated code was also exercised through the actual agent on the older electronics return window, festive-sale return window and a refund clarification. The available `shoe_return.wav` was processed after the patch: Gnani STT and TTS completed and `sample_outputs/shoe_return/reply.wav` was produced. Only one audio sample is present locally, so this does not verify unavailable organizer audio.

A fresh full qwen2.5:3b benchmark was stopped because its first prompt-only inference was still processing after several minutes. Do not describe the saved-context replays as a fresh benchmark.

## Submission items still requiring the submitter

- Record a demo video of at most two minutes showing a grounded answer, a follow-up, and a handover or unsupported question. The presenter need not appear on camera.
- Publish the repository publicly and replace the placeholder above with its URL.
- Include `demo.mp4` and the actual public repository URL in the final submission package, as required by the brief.

The draft ZIP excludes the dataset, `.env`, API keys and virtual environments. Rebuild or update it after adding the demo and final repository URL.
