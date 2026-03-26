#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
COOKBOOK_ROOT = REPO_ROOT / "tinker-cookbook"

if str(COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(COOKBOOK_ROOT))

import tinker
from tinker import types

from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Give accurate, direct answers, keep context across turns, "
    "and ask a clarifying question when the user request is ambiguous."
)


@dataclass(slots=True)
class ChatConfig:
    base_model: str
    model_path: str | None
    base_url: str | None
    renderer_name: str | None
    max_tokens: int
    temperature: float
    top_p: float
    system_prompt: str


@dataclass(slots=True)
class GenerationResult:
    assistant_text: str
    latency_ms: float
    stop_reason: str
    prompt_tokens: int
    completion_tokens: int
    renderer_name: str


def _clean_optional(value: str) -> str | None:
    value = value.strip()
    return value or None


def _resolve_renderer_name(base_model: str, renderer_override: str | None) -> str:
    if renderer_override:
        return renderer_override
    try:
        return get_recommended_renderer_name(base_model)
    except Exception as exc:
        raise ValueError(
            "Could not infer a renderer for this base model. Enter a renderer explicitly."
        ) from exc


@lru_cache(maxsize=16)
def _get_renderer_bundle(base_model: str, resolved_renderer_name: str):
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(
        resolved_renderer_name,
        tokenizer,
        model_name=base_model,
    )
    return tokenizer, renderer


class TinkerChatSession:
    def __init__(self) -> None:
        self.history: list[renderers.Message] = []

    def clear(self) -> None:
        self.history.clear()

    def export_markdown(self, config: ChatConfig) -> str:
        lines = [
            "# Tinker Chat Transcript",
            "",
            f"- Base model: `{config.base_model}`",
            f"- Model path: `{config.model_path or '(none)'}`",
            f"- Base URL: `{config.base_url or '(default)'}`",
            f"- Renderer: `{config.renderer_name or 'auto'}`",
            f"- Max tokens: `{config.max_tokens}`",
            f"- Temperature: `{config.temperature}`",
            f"- Top-p: `{config.top_p}`",
            "",
            "## System Prompt",
            "",
            config.system_prompt or "(none)",
            "",
            "## Conversation",
            "",
        ]

        for message in self.history:
            role = message["role"].capitalize()
            content = renderers.format_content_as_string(message["content"], separator="\n\n")
            lines.append(f"### {role}")
            lines.append("")
            lines.append(content or "(empty)")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def send_message(self, config: ChatConfig, user_text: str) -> GenerationResult:
        messages = self._build_prompt_messages(config.system_prompt, user_text)
        result = asyncio.run(self._generate_reply(config, messages))
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": result.assistant_text})
        return result

    def _build_prompt_messages(self, system_prompt: str, user_text: str) -> list[renderers.Message]:
        messages: list[renderers.Message] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        return messages

    async def _generate_reply(
        self,
        config: ChatConfig,
        messages: list[renderers.Message],
    ) -> GenerationResult:
        resolved_renderer_name = _resolve_renderer_name(config.base_model, config.renderer_name)
        _tokenizer, renderer = _get_renderer_bundle(config.base_model, resolved_renderer_name)

        service_client = tinker.ServiceClient(base_url=config.base_url)
        sampling_client = service_client.create_sampling_client(
            base_model=config.base_model,
            model_path=config.model_path,
        )

        model_input = renderer.build_generation_prompt(messages)
        started = time.perf_counter()
        response = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=renderer.get_stop_sequences(),
            ),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        sequence = response.sequences[0]
        parsed_message, parse_success = renderer.parse_response(sequence.tokens)
        if parse_success:
            assistant_text = renderers.format_content_as_string(
                parsed_message["content"],
                separator="\n\n",
            )
        else:
            decoded = getattr(renderer, "tokenizer", None)
            if decoded is not None:
                assistant_text = decoded.decode(sequence.tokens, skip_special_tokens=False).strip()
            else:
                assistant_text = ""
            if not assistant_text:
                assistant_text = "<empty response>"

        return GenerationResult(
            assistant_text=assistant_text,
            latency_ms=latency_ms,
            stop_reason=sequence.stop_reason,
            prompt_tokens=model_input.length,
            completion_tokens=len(sequence.tokens),
            renderer_name=resolved_renderer_name,
        )


class TinkerChatGUI:
    def __init__(self, root: tk.Tk, defaults: ChatConfig) -> None:
        self.root = root
        self.root.title("Tinker Chatbot")
        self.root.minsize(980, 760)

        self.session = TinkerChatSession()
        self.worker: threading.Thread | None = None

        self.base_model_var = tk.StringVar(value=defaults.base_model)
        self.model_path_var = tk.StringVar(value=defaults.model_path or "")
        self.base_url_var = tk.StringVar(value=defaults.base_url or "")
        self.renderer_var = tk.StringVar(value=defaults.renderer_name or "")
        self.max_tokens_var = tk.StringVar(value=str(defaults.max_tokens))
        self.temperature_var = tk.StringVar(value=str(defaults.temperature))
        self.top_p_var = tk.StringVar(value=str(defaults.top_p))
        self.status_var = tk.StringVar(
            value="Ready. Enter a prompt and press Send once TINKER_API_KEY is set."
        )

        self._build_layout(defaults.system_prompt)
        self._append_transcript(
            "system",
            "Chat session ready. Messages are sent to Tinker with the current settings.",
        )

    def _build_layout(self, system_prompt: str) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        container.grid_rowconfigure(2, weight=1)
        container.grid_columnconfigure(0, weight=1)

        settings_frame = ttk.LabelFrame(container, text="Model Settings", padding=12)
        settings_frame.grid(row=0, column=0, sticky="ew")
        settings_frame.grid_columnconfigure(1, weight=1)
        settings_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(settings_frame, text="Base model").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(settings_frame, textvariable=self.base_model_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 12)
        )
        ttk.Label(settings_frame, text="Max tokens").grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Spinbox(
            settings_frame,
            from_=32,
            to=8192,
            increment=32,
            textvariable=self.max_tokens_var,
            width=10,
        ).grid(row=0, column=3, sticky="w")

        ttk.Label(settings_frame, text="Model path").grid(row=1, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(settings_frame, textvariable=self.model_path_var).grid(
            row=1, column=1, sticky="ew", padx=(0, 12)
        )
        ttk.Label(settings_frame, text="Temperature").grid(
            row=1, column=2, sticky="w", padx=(0, 8)
        )
        ttk.Entry(settings_frame, textvariable=self.temperature_var, width=10).grid(
            row=1, column=3, sticky="w"
        )

        ttk.Label(settings_frame, text="Base URL").grid(row=2, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(settings_frame, textvariable=self.base_url_var).grid(
            row=2, column=1, sticky="ew", padx=(0, 12)
        )
        ttk.Label(settings_frame, text="Top-p").grid(row=2, column=2, sticky="w", padx=(0, 8))
        ttk.Entry(settings_frame, textvariable=self.top_p_var, width=10).grid(
            row=2, column=3, sticky="w"
        )

        ttk.Label(settings_frame, text="Renderer").grid(row=3, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(settings_frame, textvariable=self.renderer_var).grid(
            row=3, column=1, sticky="ew", padx=(0, 12)
        )
        ttk.Label(
            settings_frame,
            text="Leave renderer empty to auto-select from the base model.",
        ).grid(row=3, column=2, columnspan=2, sticky="w")

        prompt_frame = ttk.LabelFrame(container, text="System Prompt", padding=12)
        prompt_frame.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        prompt_frame.grid_columnconfigure(0, weight=1)

        self.system_prompt_text = ScrolledText(prompt_frame, height=5, wrap="word")
        self.system_prompt_text.grid(row=0, column=0, sticky="ew")
        self.system_prompt_text.insert("1.0", system_prompt)

        transcript_frame = ttk.LabelFrame(container, text="Conversation", padding=12)
        transcript_frame.grid(row=2, column=0, sticky="nsew")
        transcript_frame.grid_rowconfigure(0, weight=1)
        transcript_frame.grid_columnconfigure(0, weight=1)

        self.transcript = ScrolledText(transcript_frame, wrap="word", state="disabled")
        self.transcript.grid(row=0, column=0, sticky="nsew")
        self.transcript.tag_configure("user_name", foreground="#0b5394", font=("TkDefaultFont", 10, "bold"))
        self.transcript.tag_configure(
            "assistant_name",
            foreground="#38761d",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.transcript.tag_configure("system_name", foreground="#666666", font=("TkDefaultFont", 10, "bold"))
        self.transcript.tag_configure("body", spacing3=10)

        input_frame = ttk.LabelFrame(container, text="Your Message", padding=12)
        input_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_text = ScrolledText(input_frame, height=5, wrap="word")
        self.input_text.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.input_text.bind("<Command-Return>", self._on_send_shortcut)
        self.input_text.bind("<Control-Return>", self._on_send_shortcut)

        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=2, sticky="e", pady=(10, 0))

        self.new_chat_button = ttk.Button(input_frame, text="New Chat", command=self.new_chat)
        self.new_chat_button.grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.export_button = ttk.Button(input_frame, text="Export", command=self.export_chat)
        self.export_button.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        status_bar = ttk.Label(container, textvariable=self.status_var, anchor="w")
        status_bar.grid(row=4, column=0, sticky="ew", pady=(10, 0))

    def _on_send_shortcut(self, _event: tk.Event) -> str:
        self.send_message()
        return "break"

    def _append_transcript(self, role: str, content: str) -> None:
        tag_name = {
            "user": "user_name",
            "assistant": "assistant_name",
            "system": "system_name",
        }.get(role, "system_name")
        label = {
            "user": "You",
            "assistant": "Assistant",
            "system": "System",
        }.get(role, role.capitalize())

        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"{label}\n", tag_name)
        self.transcript.insert("end", content.strip() + "\n\n", "body")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.send_button.configure(state=state)
        self.new_chat_button.configure(state=state)
        self.export_button.configure(state=state)
        self.input_text.configure(state=state)

    def _read_input_text(self) -> str:
        return self.input_text.get("1.0", "end-1c").strip()

    def _clear_input(self) -> None:
        self.input_text.delete("1.0", "end")

    def _build_config(self) -> ChatConfig:
        base_model = self.base_model_var.get().strip()
        if not base_model:
            raise ValueError("Base model is required.")

        max_tokens = int(self.max_tokens_var.get().strip())
        if max_tokens <= 0:
            raise ValueError("Max tokens must be greater than 0.")

        temperature = float(self.temperature_var.get().strip())
        if temperature < 0:
            raise ValueError("Temperature must be 0 or greater.")

        top_p = float(self.top_p_var.get().strip())
        if not 0 < top_p <= 1:
            raise ValueError("Top-p must be between 0 and 1.")

        return ChatConfig(
            base_model=base_model,
            model_path=_clean_optional(self.model_path_var.get()),
            base_url=_clean_optional(self.base_url_var.get()),
            renderer_name=_clean_optional(self.renderer_var.get()),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=self.system_prompt_text.get("1.0", "end-1c").strip(),
        )

    def send_message(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        user_text = self._read_input_text()
        if not user_text:
            return

        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Settings", str(exc), parent=self.root)
            return

        self.status_var.set("Generating response...")
        self._set_busy(True)

        def worker() -> None:
            try:
                result = self.session.send_message(config, user_text)
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(0, self._on_send_failure, exc, tb)
                return
            self.root.after(0, self._on_send_success, config, user_text, result)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _on_send_success(
        self,
        _config: ChatConfig,
        user_text: str,
        result: GenerationResult,
    ) -> None:
        self._append_transcript("user", user_text)
        self._append_transcript("assistant", result.assistant_text)
        self._clear_input()
        self._set_busy(False)
        self.status_var.set(
            "Ready. "
            f"Renderer={result.renderer_name} | "
            f"Stop={result.stop_reason} | "
            f"Prompt={result.prompt_tokens} tok | "
            f"Completion={result.completion_tokens} tok | "
            f"{result.latency_ms:.0f} ms"
        )

    def _on_send_failure(self, exc: Exception, traceback_text: str) -> None:
        self._set_busy(False)
        self.status_var.set(f"Error: {exc}")
        messagebox.showerror(
            "Generation Failed",
            f"{exc}\n\n{traceback_text}",
            parent=self.root,
        )

    def new_chat(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        self.session.clear()
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self._append_transcript(
            "system",
            "New chat started. Current system prompt and settings will be used for the next turn.",
        )
        self.status_var.set("Ready. Conversation cleared.")

    def export_chat(self) -> None:
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Settings", str(exc), parent=self.root)
            return

        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Chat Transcript",
            defaultextension=".md",
            initialdir=str(REPO_ROOT / "chatbot"),
            initialfile="tinker-chat-transcript.md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if not file_path:
            return

        Path(file_path).write_text(self.session.export_markdown(config), encoding="utf-8")
        self.status_var.set(f"Transcript exported to {file_path}")


def parse_args() -> ChatConfig:
    parser = argparse.ArgumentParser(description="Desktop Tinker chatbot GUI")
    parser.add_argument(
        "--base-model",
        default=os.environ.get("TINKER_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        help="Base model name used for tokenizer selection and sampling.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("TINKER_MODEL_PATH"),
        help="Optional Tinker model path for custom weights or checkpoints.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TINKER_BASE_URL"),
        help="Optional Tinker base URL override.",
    )
    parser.add_argument(
        "--renderer",
        default=os.environ.get("TINKER_RENDERER"),
        help="Optional renderer override. If omitted, the app auto-selects one.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("TINKER_MAX_TOKENS", "512")),
        help="Maximum completion tokens per turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TINKER_TEMPERATURE", "0.7")),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.environ.get("TINKER_TOP_P", "0.9")),
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("TINKER_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="Initial system prompt shown in the GUI.",
    )
    args = parser.parse_args()

    return ChatConfig(
        base_model=args.base_model,
        model_path=args.model_path,
        base_url=args.base_url,
        renderer_name=args.renderer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
    )


def main() -> None:
    defaults = parse_args()
    root = tk.Tk()
    app = TinkerChatGUI(root, defaults)
    root.after(50, app.input_text.focus_set)
    root.mainloop()


if __name__ == "__main__":
    main()
