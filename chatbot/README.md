# Tinker Chatbot

This folder contains a desktop Tinker chatbot with a classic back-and-forth chat UI built on Tkinter.

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
  --top-p 0.95
```

## Notes

- `base_model` is required because the app uses it to choose the tokenizer and default renderer.
- `model_path` is optional and is meant for custom Tinker checkpoints or published weights.
- If the app cannot infer a renderer for your model, fill in the `Renderer` field manually.
- Press `Cmd+Enter` or `Ctrl+Enter` to send from the message box.
