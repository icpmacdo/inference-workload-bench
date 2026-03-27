# Tinker Chatbot

This folder contains a desktop Tinker chatbot with a classic back-and-forth chat UI built on Tkinter.

## Highlights

- Two-pane layout with a richer composer, model controls, and system prompt editing.
- `Enter` sends, `Shift+Enter` inserts a newline, `Cmd/Ctrl+S` saves a session, `Cmd/Ctrl+O` loads one, and `F5` regenerates the last reply.
- Local state auto-restores your draft, settings, and transcript between launches, even if a send is in flight.
- The transcript shows timestamps, auto-scrolls to the latest turn, and surfaces a visible assistant thinking state while replies are generating.
- Empty chats include starter prompts so the composer never opens to a completely blank state.
- Default `max_tokens` is now `5120`, and auto renderer selection prefers non-thinking variants when the model supports them.
- Save/load full sessions as JSON, export transcripts as Markdown, copy the last reply, or copy the entire transcript.
- Sampling controls now include `top_k` and `seed`, plus quick presets for balanced, precise, and creative runs.

## What It Needs

- Python 3.11+
- `TINKER_API_KEY` exported in your shell
- The reference cookbook folder kept next to this one at `../tinker-cookbook`

## Install

```bash
cd /Users/ianmacdonald/code/inference-workload-bench
python -m pip install -r chatbot/requirements.txt
```

## Run

```bash
export TINKER_API_KEY=your_api_key_here
python chatbot/tinker_chat_gui.py
```

By default, the GUI starts with `Qwen/Qwen3.5-4B`.

Optional startup overrides:

```bash
python chatbot/tinker_chat_gui.py \
  --base-model Qwen/Qwen3.5-4B \
  --max-tokens 768 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 40 \
  --seed 7
```

## Notes

- `base_model` is required because the app uses it to choose the tokenizer and default renderer.
- `model_path` is optional and is meant for custom Tinker checkpoints or published weights.
- If the app cannot infer a renderer for your model, fill in the `Renderer` field manually.
- The app restores the last local draft/session automatically. Use `--no-restore-state` to start fresh.
- Press `Enter` to send and `Shift+Enter` to add a newline in the message box. `Cmd+Enter` and `Ctrl+Enter` also send.
- Starter prompts load directly into the composer so you can edit them before sending.
