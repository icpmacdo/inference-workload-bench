#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.font import nametofont
from tkinter.scrolledtext import ScrolledText

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
COOKBOOK_ROOT = REPO_ROOT / "tinker-cookbook"
APP_STATE_PATH = REPO_ROOT / "chatbot" / ".tinker_chat_state.json"
APP_STATE_VERSION = 2

if str(COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(COOKBOOK_ROOT))

import tinker
from tinker import types

from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_names
from tinker_cookbook.tokenizer_utils import get_tokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Give accurate, direct answers, keep context across turns, "
    "and ask a clarifying question when the user request is ambiguous."
)
DEFAULT_MAX_TOKENS = 5120

MODEL_PRESETS = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "openai/gpt-oss-20b",
)

RENDERER_PRESETS = (
    "",
    "qwen3_5",
    "qwen3_5_disable_thinking",
    "qwen3",
    "qwen3_instruct",
    "llama3",
    "gpt_oss_medium_reasoning",
    "gpt_oss_no_sysprompt",
)

SAMPLING_PRESETS: dict[str, dict[str, str]] = {
    "Balanced": {"temperature": "0.7", "top_p": "0.9", "top_k": "-1"},
    "Precise": {"temperature": "0.2", "top_p": "0.85", "top_k": "40"},
    "Creative": {"temperature": "1.0", "top_p": "0.98", "top_k": "-1"},
}

SYSTEM_PROMPT_PRESETS = {
    "Default": DEFAULT_SYSTEM_PROMPT,
    "Coding Copilot": (
        "You are a pragmatic coding assistant. Ask for clarification only when required, "
        "show your reasoning briefly, and prefer concrete implementation guidance over theory."
    ),
    "Concise": (
        "You are a concise assistant. Lead with the answer, cut filler, and keep responses "
        "high signal unless the user explicitly asks for detail."
    ),
    "Research": (
        "You are a careful research assistant. State assumptions, separate facts from inference, "
        "and organize longer answers into compact, scannable sections."
    ),
}

STARTER_PROMPTS = (
    ("Plan a task", "Help me plan the next steps for this task and keep the outline concise."),
    ("Explain code", "Explain this code path, the main functions involved, and where I should edit it."),
    ("Debug an issue", "I have a bug. Help me narrow it down with the fastest debugging steps first."),
    ("Draft a reply", "Draft a direct, professional reply to this message and keep it concise."),
)

CODE_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(slots=True)
class ChatConfig:
    base_model: str
    model_path: str | None
    base_url: str | None
    renderer_name: str | None
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int | None
    system_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "model_path": self.model_path,
            "base_url": self.base_url,
            "renderer_name": self.renderer_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatConfig:
        seed = data.get("seed")
        return cls(
            base_model=str(data.get("base_model", "Qwen/Qwen3.5-4B")),
            model_path=_clean_optional(str(data.get("model_path", "") or "")),
            base_url=_clean_optional(str(data.get("base_url", "") or "")),
            renderer_name=_clean_optional(str(data.get("renderer_name", "") or "")),
            max_tokens=int(data.get("max_tokens", DEFAULT_MAX_TOKENS)),
            temperature=float(data.get("temperature", 0.7)),
            top_p=float(data.get("top_p", 0.9)),
            top_k=int(data.get("top_k", -1)),
            seed=None if seed in (None, "", "None") else int(seed),
            system_prompt=str(data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
        )


@dataclass(slots=True)
class LaunchOptions:
    defaults: ChatConfig
    cli_overrides: set[str]
    restore_state: bool


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
    created_at: str

    def to_renderer_message(self) -> renderers.Message:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        return cls(
            role=str(data["role"]),
            content=str(data["content"]),
            created_at=str(data.get("created_at") or _now_iso()),
        )


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


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_timestamp(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    return dt.strftime("%b %d, %I:%M %p").replace(" 0", " ")


def _shorten(text: str, limit: int = 72) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _collect_cli_overrides(argv: list[str]) -> set[str]:
    option_map = {
        "--base-model": "base_model",
        "--model-path": "model_path",
        "--base-url": "base_url",
        "--renderer": "renderer_name",
        "--max-tokens": "max_tokens",
        "--temperature": "temperature",
        "--top-p": "top_p",
        "--top-k": "top_k",
        "--seed": "seed",
        "--system-prompt": "system_prompt",
        "--no-restore-state": "no_restore_state",
    }
    overrides: set[str] = set()
    for token in argv:
        if token in option_map:
            overrides.add(option_map[token])
            continue
        for option, name in option_map.items():
            if token.startswith(f"{option}="):
                overrides.add(name)
                break
    return overrides


def _resolve_renderer_name(base_model: str, renderer_override: str | None) -> str:
    if renderer_override:
        return renderer_override
    try:
        recommended = get_recommended_renderer_names(base_model)
        for candidate in recommended:
            if "disable_thinking" in candidate:
                return candidate
        return recommended[0]
    except Exception as exc:
        raise ValueError(
            "Could not infer a renderer for this base model. Enter a renderer explicitly."
        ) from exc


def _sanitize_assistant_text(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).strip()


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
    def __init__(self, history: list[ChatMessage] | None = None) -> None:
        self.history: list[ChatMessage] = history or []

    def clear(self) -> None:
        self.history.clear()

    def can_regenerate(self) -> bool:
        return len(self.history) >= 2 and self.history[-2].role == "user" and self.history[-1].role == "assistant"

    def can_edit_last_turn(self) -> bool:
        return self.can_regenerate()

    def last_user_message(self) -> ChatMessage | None:
        for message in reversed(self.history):
            if message.role == "user":
                return message
        return None

    def last_assistant_message(self) -> ChatMessage | None:
        for message in reversed(self.history):
            if message.role == "assistant":
                return message
        return None

    def user_turn_count(self) -> int:
        return sum(1 for message in self.history if message.role == "user")

    def to_records(self) -> list[dict[str, str]]:
        return [message.to_dict() for message in self.history]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> TinkerChatSession:
        history = [ChatMessage.from_dict(record) for record in records]
        return cls(history=history)

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
            f"- Top-k: `{config.top_k}`",
            f"- Seed: `{config.seed if config.seed is not None else '(random)'}`",
            "",
            "## System Prompt",
            "",
            config.system_prompt or "(none)",
            "",
            "## Conversation",
            "",
        ]

        for message in self.history:
            role = message.role.capitalize()
            lines.append(f"### {role} · {_format_timestamp(message.created_at)}")
            lines.append("")
            lines.append(message.content or "(empty)")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def export_json_payload(self, config: ChatConfig) -> dict[str, Any]:
        return {
            "version": APP_STATE_VERSION,
            "saved_at": _now_iso(),
            "config": config.to_dict(),
            "history": self.to_records(),
        }

    def send_user_message(self, config: ChatConfig, user_text: str) -> GenerationResult:
        messages = self._build_prompt_messages(config.system_prompt, pending_user_text=user_text)
        result = asyncio.run(self._generate_reply(config, messages))
        self.history.append(ChatMessage(role="user", content=user_text, created_at=_now_iso()))
        self.history.append(
            ChatMessage(role="assistant", content=result.assistant_text, created_at=_now_iso())
        )
        return result

    def regenerate_last_response(self, config: ChatConfig) -> tuple[str, GenerationResult]:
        if not self.can_regenerate():
            raise ValueError("There is no completed turn to regenerate.")

        last_user = self.history[-2]
        self.history.pop()
        messages = self._build_prompt_messages(config.system_prompt)
        result = asyncio.run(self._generate_reply(config, messages))
        self.history.append(
            ChatMessage(role="assistant", content=result.assistant_text, created_at=_now_iso())
        )
        return last_user.content, result

    def pop_last_turn_to_draft(self) -> str:
        if not self.can_edit_last_turn():
            raise ValueError("There is no completed turn to move back into the composer.")

        self.history.pop()
        user_message = self.history.pop()
        return user_message.content

    def _build_prompt_messages(
        self,
        system_prompt: str,
        pending_user_text: str | None = None,
    ) -> list[renderers.Message]:
        messages: list[renderers.Message] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.extend(message.to_renderer_message() for message in self.history)
        if pending_user_text is not None:
            messages.append({"role": "user", "content": pending_user_text})
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
                seed=config.seed,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop=renderer.get_stop_sequences(),
            ),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        sequence = response.sequences[0]
        parsed_message, parse_success = renderer.parse_response(sequence.tokens)
        if parse_success:
            assistant_text = _sanitize_assistant_text(renderers.get_text_content(parsed_message))
            if not assistant_text:
                assistant_text = "<empty response>"
        else:
            decoded = getattr(renderer, "tokenizer", None)
            if decoded is not None:
                assistant_text = decoded.decode(sequence.tokens, skip_special_tokens=False).strip()
            else:
                assistant_text = ""
            assistant_text = _sanitize_assistant_text(assistant_text)
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
    def __init__(self, root: tk.Tk, launch_options: LaunchOptions) -> None:
        self.root = root
        self.launch_options = launch_options
        self.root.title("Tinker Chat")
        self.root.minsize(1140, 760)

        self.session = TinkerChatSession()
        self.worker: threading.Thread | None = None
        self.last_result: GenerationResult | None = None
        self._busy = False
        self._loading_state = False
        self._save_state_after_id: str | None = None
        self._pending_request_kind: str | None = None
        self._pending_restore_text = ""
        self._pending_user_message: ChatMessage | None = None
        self._pending_assistant_timestamp: str | None = None
        self._pending_status_text: str | None = None

        defaults = launch_options.defaults
        self.base_model_var = tk.StringVar(value=defaults.base_model)
        self.model_path_var = tk.StringVar(value=defaults.model_path or "")
        self.base_url_var = tk.StringVar(value=defaults.base_url or "")
        self.renderer_var = tk.StringVar(value=defaults.renderer_name or "")
        self.max_tokens_var = tk.StringVar(value=str(defaults.max_tokens))
        self.temperature_var = tk.StringVar(value=str(defaults.temperature))
        self.top_p_var = tk.StringVar(value=str(defaults.top_p))
        self.top_k_var = tk.StringVar(value=str(defaults.top_k))
        self.seed_var = tk.StringVar(value="" if defaults.seed is None else str(defaults.seed))
        self.status_var = tk.StringVar(
            value="Ready. Enter sends, Shift+Enter adds a newline, and sessions auto-save locally."
        )
        self.session_summary_var = tk.StringVar()
        self.usage_summary_var = tk.StringVar()
        self.renderer_hint_var = tk.StringVar()
        self.model_summary_var = tk.StringVar()
        self.draft_status_var = tk.StringVar()
        self.activity_var = tk.StringVar(value="Ready")
        self.sampling_preset_var = tk.StringVar(value="Balanced")
        self.system_prompt_preset_var = tk.StringVar(value="Default")

        self._configure_styles()
        self._build_layout(defaults.system_prompt)
        self._bind_shortcuts()
        self._bind_live_updates()
        self._restore_state_if_available()
        self._refresh_transcript(
            empty_notice=(
                "Start a conversation. Enter sends, Shift+Enter adds a newline, "
                "and your draft plus history are restored automatically next time."
            )
        )
        self._update_renderer_hint()
        self._update_sampling_preset_indicator()
        self._update_system_prompt_preset_indicator()
        self._update_session_summaries()
        self._update_draft_status()
        self._sync_button_states()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        default_font = nametofont("TkDefaultFont")
        default_font.configure(size=11)
        text_font = nametofont("TkTextFont")
        text_font.configure(size=12)
        fixed_font = nametofont("TkFixedFont")
        fixed_font.configure(size=11)

        style.configure("Title.TLabel", font=(default_font.actual("family"), 18, "bold"))
        style.configure("Subtitle.TLabel", foreground="#596780")
        style.configure("Section.TLabelframe", padding=10)
        style.configure("Section.TLabelframe.Label", font=(default_font.actual("family"), 11, "bold"))
        style.configure("Muted.TLabel", foreground="#667085")

    def _build_layout(self, system_prompt: str) -> None:
        container = ttk.Frame(self.root, padding=14)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.grid_columnconfigure(0, weight=1)

        title_block = ttk.Frame(header)
        title_block.grid(row=0, column=0, sticky="w")
        ttk.Label(title_block, text="Tinker Chat", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            title_block,
            text="Desktop client for fast multi-turn prompting, review, and session replay.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        header_right = ttk.Frame(header)
        header_right.grid(row=0, column=1, sticky="e")
        ttk.Label(header_right, textvariable=self.session_summary_var).grid(row=0, column=0, sticky="e")
        ttk.Label(header_right, textvariable=self.usage_summary_var, style="Muted.TLabel").grid(
            row=1, column=0, sticky="e", pady=(2, 0)
        )
        ttk.Label(header_right, textvariable=self.activity_var, style="Muted.TLabel").grid(
            row=2, column=0, sticky="e", pady=(6, 0)
        )
        self.progress = ttk.Progressbar(header_right, mode="indeterminate", length=120)
        self.progress.grid(row=3, column=0, sticky="e", pady=(6, 0))

        body = ttk.Panedwindow(container, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        conversation_shell = ttk.Frame(body)
        sidebar = ttk.Frame(body, padding=(12, 0, 0, 0))
        conversation_shell.grid_rowconfigure(1, weight=1)
        conversation_shell.grid_columnconfigure(0, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)
        body.add(conversation_shell, weight=4)
        body.add(sidebar, weight=2)

        transcript_frame = ttk.LabelFrame(conversation_shell, text="Conversation", style="Section.TLabelframe")
        transcript_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        transcript_frame.grid_rowconfigure(0, weight=1)
        transcript_frame.grid_columnconfigure(0, weight=1)

        self.transcript = ScrolledText(
            transcript_frame,
            wrap="word",
            state="disabled",
            padx=12,
            pady=12,
            relief="flat",
            borderwidth=0,
            background="#fbfcfe",
            foreground="#182230",
            insertbackground="#182230",
        )
        self.transcript.grid(row=0, column=0, sticky="nsew")
        self.transcript.tag_configure(
            "user_meta",
            foreground="#1d4ed8",
            font=(nametofont("TkDefaultFont").actual("family"), 10, "bold"),
            spacing1=14,
        )
        self.transcript.tag_configure(
            "assistant_meta",
            foreground="#15803d",
            font=(nametofont("TkDefaultFont").actual("family"), 10, "bold"),
            spacing1=14,
        )
        self.transcript.tag_configure(
            "notice_meta",
            foreground="#667085",
            font=(nametofont("TkDefaultFont").actual("family"), 10, "bold"),
            spacing1=14,
        )
        self.transcript.tag_configure(
            "user_body",
            background="#e8f0ff",
            lmargin1=12,
            lmargin2=12,
            rmargin=40,
            spacing1=4,
            spacing3=12,
        )
        self.transcript.tag_configure(
            "assistant_body",
            background="#eef9ef",
            lmargin1=12,
            lmargin2=12,
            rmargin=40,
            spacing1=4,
            spacing3=12,
        )
        self.transcript.tag_configure(
            "notice_body",
            background="#f2f4f7",
            foreground="#344054",
            lmargin1=12,
            lmargin2=12,
            rmargin=40,
            spacing1=4,
            spacing3=12,
        )
        self.transcript.tag_configure(
            "pending_body",
            background="#fff7e8",
            foreground="#9a6700",
            lmargin1=12,
            lmargin2=12,
            rmargin=40,
            spacing1=4,
            spacing3=12,
        )
        self.transcript.tag_configure(
            "code_block",
            background="#101828",
            foreground="#f8fafc",
            font=("TkFixedFont", 11),
            lmargin1=22,
            lmargin2=22,
            rmargin=50,
            spacing1=6,
            spacing3=6,
        )
        self.transcript.tag_configure(
            "code_label",
            foreground="#98a2b3",
            font=(nametofont("TkDefaultFont").actual("family"), 9, "bold"),
        )

        self.starters_frame = ttk.Frame(transcript_frame, padding=(0, 12, 0, 0))
        self.starters_frame.grid(row=1, column=0, sticky="ew")
        self.starters_frame.grid_columnconfigure(0, weight=1)
        self.starters_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.starters_frame, text="Starter prompts", style="Muted.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        for index, (label, prompt) in enumerate(STARTER_PROMPTS):
            row = (index // 2) + 1
            column = index % 2
            padx = (0, 6) if column == 0 else (6, 0)
            ttk.Button(
                self.starters_frame,
                text=label,
                command=lambda prompt=prompt: self._use_starter_prompt(prompt),
            ).grid(row=row, column=column, sticky="ew", padx=padx, pady=(8, 0))

        ttk.Label(
            self.starters_frame,
            text="Click a starter to load it into the composer, then press Enter to send.",
            style="Muted.TLabel",
            wraplength=640,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        composer_frame = ttk.LabelFrame(sidebar, text="Composer", style="Section.TLabelframe")
        composer_frame.grid(row=0, column=0, sticky="ew")
        composer_frame.grid_columnconfigure(0, weight=1)

        self.input_text = ScrolledText(
            composer_frame,
            height=8,
            wrap="word",
            padx=10,
            pady=10,
        )
        self.input_text.grid(row=0, column=0, columnspan=4, sticky="ew")
        self.input_text.bind("<Return>", self._on_enter_pressed)
        self.input_text.bind("<Command-Return>", self._on_send_shortcut)
        self.input_text.bind("<Control-Return>", self._on_send_shortcut)

        self.clear_draft_button = ttk.Button(composer_frame, text="Clear Draft", command=self.clear_draft)
        self.clear_draft_button.grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.edit_last_button = ttk.Button(composer_frame, text="Edit Last", command=self.edit_last_turn)
        self.edit_last_button.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        self.retry_button = ttk.Button(composer_frame, text="Retry", command=self.retry_last_response)
        self.retry_button.grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(10, 0))

        self.send_button = ttk.Button(composer_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=3, sticky="e", pady=(10, 0))

        ttk.Label(composer_frame, textvariable=self.draft_status_var, style="Muted.TLabel").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )

        session_frame = ttk.LabelFrame(sidebar, text="Session", style="Section.TLabelframe")
        session_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        session_frame.grid_columnconfigure(0, weight=1)
        session_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(session_frame, textvariable=self.model_summary_var, wraplength=340).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(session_frame, textvariable=self.usage_summary_var, style="Muted.TLabel", wraplength=340).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 10)
        )

        self.new_chat_button = ttk.Button(session_frame, text="New Chat", command=self.new_chat)
        self.new_chat_button.grid(row=2, column=0, sticky="ew")

        self.copy_last_button = ttk.Button(session_frame, text="Copy Last", command=self.copy_last_response)
        self.copy_last_button.grid(row=2, column=1, sticky="ew", padx=(8, 0))

        self.copy_transcript_button = ttk.Button(
            session_frame,
            text="Copy Transcript",
            command=self.copy_transcript,
        )
        self.copy_transcript_button.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        self.export_button = ttk.Button(session_frame, text="Export Markdown", command=self.export_chat)
        self.export_button.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        self.save_session_button = ttk.Button(session_frame, text="Save Session", command=self.save_session)
        self.save_session_button.grid(row=4, column=0, sticky="ew", pady=(8, 0))

        self.load_session_button = ttk.Button(session_frame, text="Load Session", command=self.load_session)
        self.load_session_button.grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        model_frame = ttk.LabelFrame(sidebar, text="Model & Transport", style="Section.TLabelframe")
        model_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        model_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="Base model").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.base_model_combo = ttk.Combobox(model_frame, textvariable=self.base_model_var, values=MODEL_PRESETS)
        self.base_model_combo.grid(row=0, column=1, sticky="ew")

        ttk.Label(model_frame, text="Model path").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Entry(model_frame, textvariable=self.model_path_var).grid(row=1, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(model_frame, text="Base URL").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Entry(model_frame, textvariable=self.base_url_var).grid(row=2, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(model_frame, text="Renderer").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.renderer_combo = ttk.Combobox(model_frame, textvariable=self.renderer_var, values=RENDERER_PRESETS)
        self.renderer_combo.grid(row=3, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(model_frame, textvariable=self.renderer_hint_var, style="Muted.TLabel", wraplength=340).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        sampling_frame = ttk.LabelFrame(sidebar, text="Sampling", style="Section.TLabelframe")
        sampling_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        sampling_frame.grid_columnconfigure(1, weight=1)
        sampling_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(sampling_frame, text="Preset").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.sampling_preset_combo = ttk.Combobox(
            sampling_frame,
            textvariable=self.sampling_preset_var,
            values=[*SAMPLING_PRESETS.keys(), "Custom"],
            state="readonly",
        )
        self.sampling_preset_combo.grid(row=0, column=1, sticky="ew", columnspan=3)
        self.sampling_preset_combo.bind("<<ComboboxSelected>>", self._apply_sampling_preset)

        ttk.Label(sampling_frame, text="Max tokens").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Spinbox(
            sampling_frame,
            from_=32,
            to=8192,
            increment=32,
            textvariable=self.max_tokens_var,
            width=10,
        ).grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(sampling_frame, text="Temperature").grid(
            row=1, column=2, sticky="w", padx=(12, 8), pady=(8, 0)
        )
        ttk.Entry(sampling_frame, textvariable=self.temperature_var, width=10).grid(
            row=1, column=3, sticky="w", pady=(8, 0)
        )

        ttk.Label(sampling_frame, text="Top-p").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Entry(sampling_frame, textvariable=self.top_p_var, width=10).grid(
            row=2, column=1, sticky="w", pady=(8, 0)
        )

        ttk.Label(sampling_frame, text="Top-k").grid(row=2, column=2, sticky="w", padx=(12, 8), pady=(8, 0))
        ttk.Entry(sampling_frame, textvariable=self.top_k_var, width=10).grid(
            row=2, column=3, sticky="w", pady=(8, 0)
        )

        ttk.Label(sampling_frame, text="Seed").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Entry(sampling_frame, textvariable=self.seed_var, width=10).grid(
            row=3, column=1, sticky="w", pady=(8, 0)
        )

        ttk.Label(
            sampling_frame,
            text="Use a seed for reproducible retries. Leave it blank for fresh randomness each turn.",
            style="Muted.TLabel",
            wraplength=340,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))

        prompt_frame = ttk.LabelFrame(sidebar, text="System Prompt", style="Section.TLabelframe")
        prompt_frame.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
        prompt_frame.grid_columnconfigure(1, weight=1)
        sidebar.grid_rowconfigure(4, weight=1)

        ttk.Label(prompt_frame, text="Preset").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.system_prompt_preset_combo = ttk.Combobox(
            prompt_frame,
            textvariable=self.system_prompt_preset_var,
            values=[*SYSTEM_PROMPT_PRESETS.keys(), "Custom"],
            state="readonly",
        )
        self.system_prompt_preset_combo.grid(row=0, column=1, sticky="ew")
        self.system_prompt_preset_combo.bind("<<ComboboxSelected>>", self._apply_system_prompt_preset)

        self.reset_prompt_button = ttk.Button(prompt_frame, text="Reset", command=self.reset_system_prompt)
        self.reset_prompt_button.grid(row=0, column=2, sticky="e", padx=(8, 0))

        self.system_prompt_text = ScrolledText(prompt_frame, height=10, wrap="word", padx=10, pady=10)
        self.system_prompt_text.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        self.system_prompt_text.insert("1.0", system_prompt)
        prompt_frame.grid_rowconfigure(1, weight=1)

        status_bar = ttk.Label(container, textvariable=self.status_var, anchor="w", style="Muted.TLabel")
        status_bar.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _bind_shortcuts(self) -> None:
        for sequence in ("<Command-s>", "<Control-s>"):
            self.root.bind(sequence, self._on_save_shortcut)
        for sequence in ("<Command-o>", "<Control-o>"):
            self.root.bind(sequence, self._on_open_shortcut)
        for sequence in ("<Command-n>", "<Control-n>"):
            self.root.bind(sequence, self._on_new_chat_shortcut)
        self.root.bind("<F5>", self._on_retry_shortcut)
        self.root.bind("<Escape>", self._focus_input)

    def _bind_live_updates(self) -> None:
        for variable in (
            self.base_model_var,
            self.model_path_var,
            self.base_url_var,
            self.renderer_var,
            self.max_tokens_var,
            self.temperature_var,
            self.top_p_var,
            self.top_k_var,
            self.seed_var,
        ):
            variable.trace_add("write", self._on_settings_changed)

        self.input_text.bind("<<Modified>>", self._on_input_modified)
        self.system_prompt_text.bind("<<Modified>>", self._on_system_prompt_modified)

    def _on_settings_changed(self, *_args: object) -> None:
        self._update_renderer_hint()
        self._update_sampling_preset_indicator()
        self._update_session_summaries()
        self._sync_button_states()
        self._schedule_state_save()

    def _on_input_modified(self, _event: tk.Event) -> None:
        if not self.input_text.edit_modified():
            return
        self.input_text.edit_modified(False)
        self._update_draft_status()
        self._sync_button_states()
        self._schedule_state_save()

    def _on_system_prompt_modified(self, _event: tk.Event) -> None:
        if not self.system_prompt_text.edit_modified():
            return
        self.system_prompt_text.edit_modified(False)
        self._update_system_prompt_preset_indicator()
        self._schedule_state_save()

    def _on_enter_pressed(self, _event: tk.Event) -> str:
        if _event.state & 0x0001:
            self.input_text.insert("insert", "\n")
            return "break"
        self.send_message()
        return "break"

    def _on_shift_enter_pressed(self, _event: tk.Event) -> str:
        self.input_text.insert("insert", "\n")
        return "break"

    def _on_send_shortcut(self, _event: tk.Event) -> str:
        self.send_message()
        return "break"

    def _on_save_shortcut(self, _event: tk.Event) -> str:
        self.save_session()
        return "break"

    def _on_open_shortcut(self, _event: tk.Event) -> str:
        self.load_session()
        return "break"

    def _on_new_chat_shortcut(self, _event: tk.Event) -> str:
        self.new_chat()
        return "break"

    def _on_retry_shortcut(self, _event: tk.Event) -> str:
        self.retry_last_response()
        return "break"

    def _focus_input(self, _event: tk.Event | None = None) -> str:
        self.input_text.focus_set()
        return "break"

    def _read_input_text(self, *, strip: bool = True) -> str:
        text = self.input_text.get("1.0", "end-1c")
        return text.strip() if strip else text

    def _read_system_prompt_text(self) -> str:
        return self.system_prompt_text.get("1.0", "end-1c").strip()

    def _set_text_widget_value(self, widget: ScrolledText, text: str) -> None:
        previous_state = str(widget.cget("state"))
        if previous_state == "disabled":
            widget.configure(state="normal")
        widget.delete("1.0", "end")
        if text:
            widget.insert("1.0", text)
        widget.edit_modified(False)
        if previous_state == "disabled":
            widget.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
            if self._pending_status_text is None:
                self.activity_var.set("Ready")
        self._sync_button_states()

    def _sync_button_states(self) -> None:
        draft_present = bool(self._read_input_text())
        has_history = bool(self.session.history)
        can_retry = self.session.can_regenerate()
        can_edit = self.session.can_edit_last_turn()
        has_last_response = self.session.last_assistant_message() is not None

        self.send_button.configure(state="disabled" if self._busy or not draft_present else "normal")
        self.retry_button.configure(state="disabled" if self._busy or not can_retry else "normal")
        self.edit_last_button.configure(state="disabled" if self._busy or not can_edit else "normal")
        self.new_chat_button.configure(state="disabled" if self._busy else "normal")
        self.clear_draft_button.configure(state="disabled" if self._busy or not draft_present else "normal")
        self.copy_last_button.configure(state="disabled" if not has_last_response else "normal")
        self.copy_transcript_button.configure(state="disabled" if not has_history else "normal")
        self.export_button.configure(state="disabled" if not has_history else "normal")
        self.save_session_button.configure(state="disabled" if self._busy or not has_history else "normal")
        self.load_session_button.configure(state="disabled" if self._busy else "normal")
        state = "disabled" if self._busy else "normal"
        self.input_text.configure(state=state)

    def _update_renderer_hint(self) -> None:
        base_model = self.base_model_var.get().strip()
        renderer_override = _clean_optional(self.renderer_var.get())
        if not base_model:
            self.renderer_hint_var.set("Enter a base model. Leave renderer blank to auto-select the best match.")
            return

        try:
            recommended = get_recommended_renderer_names(base_model)
        except Exception:
            if renderer_override:
                self.renderer_hint_var.set(
                    f"Manual renderer: {renderer_override}. This model is not in the local registry, so auto-matching is unavailable."
                )
            else:
                self.renderer_hint_var.set(
                    "Model not recognized by the local registry. Enter a renderer manually if generation fails."
                )
            return

        if renderer_override:
            self.renderer_hint_var.set(
                f"Manual renderer: {renderer_override}. Recommended: {', '.join(recommended)}."
            )
        else:
            auto_renderer = _resolve_renderer_name(base_model, None)
            alternates = [candidate for candidate in recommended if candidate != auto_renderer]
            self.renderer_hint_var.set(
                f"Auto renderer: {auto_renderer}. Alternate matches: {', '.join(alternates) or 'none'}."
            )

    def _update_sampling_preset_indicator(self) -> None:
        current = {
            "temperature": self.temperature_var.get().strip(),
            "top_p": self.top_p_var.get().strip(),
            "top_k": self.top_k_var.get().strip(),
        }
        for name, values in SAMPLING_PRESETS.items():
            if current == values:
                self.sampling_preset_var.set(name)
                return
        self.sampling_preset_var.set("Custom")

    def _update_system_prompt_preset_indicator(self) -> None:
        prompt = self.system_prompt_text.get("1.0", "end-1c").strip()
        for name, value in SYSTEM_PROMPT_PRESETS.items():
            if prompt == value:
                self.system_prompt_preset_var.set(name)
                return
        self.system_prompt_preset_var.set("Custom")

    def _update_draft_status(self) -> None:
        draft = self._read_input_text(strip=False)
        stripped = draft.strip()
        char_count = len(stripped)
        line_count = max(1, stripped.count("\n") + 1) if stripped else 1
        self.draft_status_var.set(
            f"{char_count} chars • {line_count} lines • Enter sends • Shift+Enter newline • Cmd/Ctrl+S saves session"
        )

    def _update_session_summaries(self) -> None:
        turns = self.session.user_turn_count()
        message_count = len(self.session.history)
        model = self.base_model_var.get().strip() or "(no model)"
        self.session_summary_var.set(f"{turns} turns • {message_count} messages")
        self.model_summary_var.set(f"Current model: {model}")
        if self.last_result is None:
            self.usage_summary_var.set("No responses yet in this run.")
        else:
            self.usage_summary_var.set(
                f"Last reply: {self.last_result.latency_ms:.0f} ms • "
                f"{self.last_result.prompt_tokens} prompt tok • "
                f"{self.last_result.completion_tokens} completion tok • "
                f"{self.last_result.renderer_name}"
            )

    def _refresh_transcript(self, empty_notice: str | None = None) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        has_transcript_content = bool(
            self.session.history or self._pending_user_message or self._pending_status_text
        )
        if not has_transcript_content:
            self._insert_notice(empty_notice or "New chat ready.")
        else:
            for message in self.session.history:
                self._insert_message(message)
            if self._pending_user_message is not None:
                self._insert_message(self._pending_user_message)
            self._insert_pending_reply()
        self.transcript.configure(state="disabled")
        self.transcript.see("end")
        self._update_starter_visibility()

    def _append_to_transcript(self, message: ChatMessage) -> None:
        self.transcript.configure(state="normal")
        self._insert_message(message)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _append_notice(self, notice: str) -> None:
        self.transcript.configure(state="normal")
        self._insert_notice(notice)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _insert_notice(self, notice: str) -> None:
        self.transcript.insert("end", "System\n", ("notice_meta",))
        self.transcript.insert("end", notice.strip() + "\n\n", ("notice_body",))

    def _insert_pending_reply(self) -> None:
        if self._pending_status_text is None or self._pending_assistant_timestamp is None:
            return
        self.transcript.insert(
            "end",
            f"Assistant • {_format_timestamp(self._pending_assistant_timestamp)}\n",
            ("assistant_meta",),
        )
        self.transcript.insert("end", self._pending_status_text + "\n\n", ("pending_body",))

    def _insert_message(self, message: ChatMessage) -> None:
        role_label = {
            "user": "You",
            "assistant": "Assistant",
        }.get(message.role, message.role.capitalize())
        meta_tag = {
            "user": "user_meta",
            "assistant": "assistant_meta",
        }.get(message.role, "notice_meta")
        body_tag = {
            "user": "user_body",
            "assistant": "assistant_body",
        }.get(message.role, "notice_body")

        self.transcript.insert(
            "end",
            f"{role_label} • {_format_timestamp(message.created_at)}\n",
            (meta_tag,),
        )
        self._insert_formatted_content(message.content or "(empty)", body_tag)
        self.transcript.insert("end", "\n\n")

    def _insert_formatted_content(self, content: str, body_tag: str) -> None:
        last_end = 0
        matched = False
        for match in CODE_BLOCK_RE.finditer(content):
            matched = True
            if match.start() > last_end:
                self.transcript.insert("end", content[last_end : match.start()], (body_tag,))
            language = match.group(1).strip()
            code_text = match.group(2).rstrip("\n")
            if language:
                self.transcript.insert("end", f"{language}\n", ("code_label",))
            self.transcript.insert("end", code_text or "(empty code block)", ("code_block",))
            self.transcript.insert("end", "\n", (body_tag,))
            last_end = match.end()

        if not matched or last_end < len(content):
            self.transcript.insert("end", content[last_end:], (body_tag,))

    def _update_starter_visibility(self) -> None:
        show_starters = not (
            self.session.history or self._pending_user_message or self._pending_status_text
        )
        if show_starters:
            self.starters_frame.grid()
        else:
            self.starters_frame.grid_remove()

    def _use_starter_prompt(self, prompt: str) -> None:
        if self._busy:
            return
        self._set_text_widget_value(self.input_text, prompt)
        self.status_var.set("Starter prompt loaded into the composer. Edit it or press Enter to send.")
        self._update_draft_status()
        self._sync_button_states()
        self._schedule_state_save()
        self.input_text.focus_set()

    def _begin_pending_request(self, kind: str, user_text: str | None = None) -> None:
        self._pending_request_kind = kind
        self._pending_restore_text = user_text or ""
        self._pending_user_message = (
            ChatMessage(role="user", content=user_text, created_at=_now_iso())
            if kind == "send" and user_text is not None
            else None
        )
        self._pending_assistant_timestamp = _now_iso()
        self._pending_status_text = (
            "Thinking…"
            if kind == "send"
            else "Regenerating the last reply…"
        )
        self.activity_var.set(self._pending_status_text)
        if kind == "send" and user_text is not None:
            self._set_text_widget_value(self.input_text, "")
        self._refresh_transcript()
        self._update_draft_status()
        self._sync_button_states()

    def _clear_pending_request(self, *, restore_draft: bool = False) -> None:
        restore_text = (
            self._pending_restore_text
            if restore_draft and self._pending_request_kind == "send"
            else ""
        )
        self._pending_request_kind = None
        self._pending_restore_text = ""
        self._pending_user_message = None
        self._pending_assistant_timestamp = None
        self._pending_status_text = None
        self.activity_var.set("Ready")
        if restore_text:
            self._set_text_widget_value(self.input_text, restore_text)
        self._refresh_transcript()

    def _schedule_state_save(self) -> None:
        if self._loading_state:
            return
        if self._save_state_after_id is not None:
            self.root.after_cancel(self._save_state_after_id)
        self._save_state_after_id = self.root.after(350, self._save_state)

    def _capture_ui_state(self) -> dict[str, Any]:
        return {
            "version": APP_STATE_VERSION,
            "saved_at": _now_iso(),
            "geometry": self.root.geometry(),
            "settings": {
                "base_model": self.base_model_var.get(),
                "model_path": self.model_path_var.get(),
                "base_url": self.base_url_var.get(),
                "renderer_name": self.renderer_var.get(),
                "max_tokens": self.max_tokens_var.get(),
                "temperature": self.temperature_var.get(),
                "top_p": self.top_p_var.get(),
                "top_k": self.top_k_var.get(),
                "seed": self.seed_var.get(),
                "system_prompt": self.system_prompt_text.get("1.0", "end-1c"),
            },
            "draft": self._draft_text_for_persistence(),
            "history": self.session.to_records(),
        }

    def _draft_text_for_persistence(self) -> str:
        draft = self._read_input_text(strip=False)
        if draft:
            return draft
        if self._pending_request_kind == "send":
            return self._pending_restore_text
        return ""

    def _save_state(self) -> None:
        self._save_state_after_id = None
        try:
            APP_STATE_PATH.write_text(
                json.dumps(self._capture_ui_state(), indent=2),
                encoding="utf-8",
            )
        except Exception:
            # Local state should never interrupt the main workflow.
            pass

    def _restore_state_if_available(self) -> None:
        if not self.launch_options.restore_state or not APP_STATE_PATH.exists():
            return

        try:
            payload = json.loads(APP_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            self.status_var.set("Ready. Previous local state could not be restored, so the app started fresh.")
            return

        settings = payload.get("settings", {})
        history = payload.get("history", [])
        draft = payload.get("draft", "")
        geometry = payload.get("geometry")
        state_version = int(payload.get("version", 1))
        overrides = self.launch_options.cli_overrides

        self._loading_state = True
        try:
            if "base_model" not in overrides:
                self.base_model_var.set(str(settings.get("base_model", self.base_model_var.get())))
            if "model_path" not in overrides:
                self.model_path_var.set(str(settings.get("model_path", self.model_path_var.get())))
            if "base_url" not in overrides:
                self.base_url_var.set(str(settings.get("base_url", self.base_url_var.get())))
            if "renderer_name" not in overrides:
                self.renderer_var.set(str(settings.get("renderer_name", self.renderer_var.get())))
            if "max_tokens" not in overrides:
                restored_max_tokens = settings.get("max_tokens", self.max_tokens_var.get())
                if state_version < 2 and str(restored_max_tokens).strip() == "512":
                    restored_max_tokens = str(DEFAULT_MAX_TOKENS)
                self.max_tokens_var.set(str(restored_max_tokens))
            if "temperature" not in overrides:
                self.temperature_var.set(str(settings.get("temperature", self.temperature_var.get())))
            if "top_p" not in overrides:
                self.top_p_var.set(str(settings.get("top_p", self.top_p_var.get())))
            if "top_k" not in overrides:
                self.top_k_var.set(str(settings.get("top_k", self.top_k_var.get())))
            if "seed" not in overrides:
                saved_seed = settings.get("seed", self.seed_var.get())
                self.seed_var.set("" if saved_seed in (None, "None") else str(saved_seed))
            if "system_prompt" not in overrides:
                self._set_text_widget_value(
                    self.system_prompt_text,
                    str(settings.get("system_prompt", self._read_system_prompt_text())),
                )

            if draft:
                self._set_text_widget_value(self.input_text, str(draft))

            if history:
                self.session = TinkerChatSession.from_records(history)

            if geometry:
                try:
                    self.root.geometry(str(geometry))
                except tk.TclError:
                    pass
        finally:
            self._loading_state = False

        if self.session.history or draft:
            self.status_var.set("Restored the last local session. Enter sends, Shift+Enter adds a newline.")

    def _build_config(self) -> ChatConfig:
        base_model = self.base_model_var.get().strip()
        if not base_model:
            raise ValueError("Base model is required.")

        try:
            max_tokens = int(self.max_tokens_var.get().strip())
        except ValueError as exc:
            raise ValueError("Max tokens must be an integer.") from exc
        if max_tokens <= 0:
            raise ValueError("Max tokens must be greater than 0.")

        try:
            temperature = float(self.temperature_var.get().strip())
        except ValueError as exc:
            raise ValueError("Temperature must be a number.") from exc
        if temperature < 0:
            raise ValueError("Temperature must be 0 or greater.")

        try:
            top_p = float(self.top_p_var.get().strip())
        except ValueError as exc:
            raise ValueError("Top-p must be a number.") from exc
        if not 0 < top_p <= 1:
            raise ValueError("Top-p must be between 0 and 1.")

        try:
            top_k = int(self.top_k_var.get().strip())
        except ValueError as exc:
            raise ValueError("Top-k must be an integer.") from exc
        if top_k == 0 or top_k < -1:
            raise ValueError("Top-k must be -1 (disabled) or a positive integer.")

        seed_value = self.seed_var.get().strip()
        try:
            seed = None if not seed_value else int(seed_value)
        except ValueError as exc:
            raise ValueError("Seed must be blank or an integer.") from exc

        return ChatConfig(
            base_model=base_model,
            model_path=_clean_optional(self.model_path_var.get()),
            base_url=_clean_optional(self.base_url_var.get()),
            renderer_name=_clean_optional(self.renderer_var.get()),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            system_prompt=self._read_system_prompt_text(),
        )

    def clear_draft(self) -> None:
        if not self._read_input_text():
            return
        self._set_text_widget_value(self.input_text, "")
        self.status_var.set("Draft cleared.")
        self._update_draft_status()
        self._sync_button_states()
        self._schedule_state_save()

    def send_message(self) -> None:
        if self._busy:
            return

        user_text = self._read_input_text()
        if not user_text:
            return
        if not os.environ.get("TINKER_API_KEY"):
            messagebox.showerror(
                "Missing API Key",
                "Set TINKER_API_KEY before sending a message.",
                parent=self.root,
            )
            return

        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Settings", str(exc), parent=self.root)
            return

        self._begin_pending_request("send", user_text)
        self.status_var.set(f"Generating response for: {_shorten(user_text)}")
        self._set_busy(True)
        self._schedule_state_save()

        def worker() -> None:
            try:
                result = self.session.send_user_message(config, user_text)
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(0, self._on_request_failure, exc, tb)
                return
            self.root.after(0, self._on_send_success, user_text, result)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def retry_last_response(self) -> None:
        if self._busy or not self.session.can_regenerate():
            return

        if not os.environ.get("TINKER_API_KEY"):
            messagebox.showerror(
                "Missing API Key",
                "Set TINKER_API_KEY before retrying a message.",
                parent=self.root,
            )
            return

        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Settings", str(exc), parent=self.root)
            return

        self._begin_pending_request("retry")
        self.status_var.set("Regenerating the last assistant reply…")
        self._set_busy(True)

        def worker() -> None:
            try:
                user_text, result = self.session.regenerate_last_response(config)
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(0, self._on_request_failure, exc, tb)
                return
            self.root.after(0, self._on_retry_success, user_text, result)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def edit_last_turn(self) -> None:
        if self._busy:
            return
        try:
            draft = self.session.pop_last_turn_to_draft()
        except Exception as exc:
            messagebox.showerror("Edit Last Turn", str(exc), parent=self.root)
            return

        self._set_text_widget_value(self.input_text, draft)
        self._refresh_transcript(
            empty_notice=(
                "The last completed turn was moved back into the composer. "
                "Edit it, then press Enter to send again."
            )
        )
        self.status_var.set("Moved the last completed turn back into the composer.")
        self._update_session_summaries()
        self._update_draft_status()
        self._sync_button_states()
        self._schedule_state_save()
        self.input_text.focus_set()

    def _on_send_success(self, user_text: str, result: GenerationResult) -> None:
        self.last_result = result
        self._clear_pending_request()
        self._set_busy(False)
        self._update_session_summaries()
        self._update_draft_status()
        self.status_var.set(
            f"Ready. Sent: {_shorten(user_text)} • {result.latency_ms:.0f} ms • "
            f"{result.prompt_tokens}/{result.completion_tokens} tok • stop={result.stop_reason}"
        )
        self._schedule_state_save()
        self.input_text.focus_set()

    def _on_retry_success(self, user_text: str, result: GenerationResult) -> None:
        self.last_result = result
        self._clear_pending_request()
        self._set_busy(False)
        self._update_session_summaries()
        self.status_var.set(
            f"Ready. Regenerated: {_shorten(user_text)} • {result.latency_ms:.0f} ms • "
            f"{result.prompt_tokens}/{result.completion_tokens} tok"
        )
        self._update_draft_status()
        self._schedule_state_save()
        self.input_text.focus_set()

    def _on_request_failure(self, exc: Exception, traceback_text: str) -> None:
        restore_draft = self._pending_request_kind == "send"
        self._clear_pending_request(restore_draft=restore_draft)
        self._set_busy(False)
        self._update_draft_status()
        self.status_var.set(f"Error: {exc}")
        self._schedule_state_save()
        messagebox.showerror(
            "Generation Failed",
            f"{exc}\n\n{traceback_text}",
            parent=self.root,
        )

    def new_chat(self) -> None:
        if self._busy:
            return

        has_work = bool(self.session.history or self._read_input_text())
        if has_work and not messagebox.askyesno(
            "Start New Chat",
            "Clear the current conversation and draft?",
            parent=self.root,
        ):
            return

        self.session.clear()
        self.last_result = None
        self._set_text_widget_value(self.input_text, "")
        self._refresh_transcript(
            empty_notice=(
                "New chat ready. Current settings and system prompt will be used on the next turn."
            )
        )
        self._update_session_summaries()
        self._update_draft_status()
        self._sync_button_states()
        self.status_var.set("Ready. Conversation cleared.")
        self._schedule_state_save()
        self.input_text.focus_set()

    def copy_last_response(self) -> None:
        message = self.session.last_assistant_message()
        if message is None:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(message.content)
        self.status_var.set("Copied the last assistant response to the clipboard.")

    def copy_transcript(self) -> None:
        try:
            config = self._build_config()
        except Exception:
            config = self.launch_options.defaults
        transcript = self.session.export_markdown(config)
        self.root.clipboard_clear()
        self.root.clipboard_append(transcript)
        self.status_var.set("Copied the full transcript to the clipboard.")

    def export_chat(self) -> None:
        if not self.session.history:
            return

        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Settings", str(exc), parent=self.root)
            return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Chat Transcript",
            defaultextension=".md",
            initialdir=str(REPO_ROOT / "chatbot"),
            initialfile=f"tinker-chat-{timestamp}.md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if not file_path:
            return

        Path(file_path).write_text(self.session.export_markdown(config), encoding="utf-8")
        self.status_var.set(f"Transcript exported to {file_path}")

    def save_session(self) -> None:
        if not self.session.history:
            return

        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Settings", str(exc), parent=self.root)
            return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Session",
            defaultextension=".json",
            initialdir=str(REPO_ROOT / "chatbot"),
            initialfile=f"tinker-chat-session-{timestamp}.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        Path(file_path).write_text(
            json.dumps(self.session.export_json_payload(config), indent=2),
            encoding="utf-8",
        )
        self.status_var.set(f"Session saved to {file_path}")

    def load_session(self) -> None:
        if self._busy:
            return

        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Load Session",
            initialdir=str(REPO_ROOT / "chatbot"),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
            config = ChatConfig.from_dict(payload["config"])
            session = TinkerChatSession.from_records(payload.get("history", []))
        except Exception as exc:
            messagebox.showerror(
                "Load Session Failed",
                f"Could not load session: {exc}",
                parent=self.root,
            )
            return

        if (self.session.history or self._read_input_text()) and not messagebox.askyesno(
            "Load Session",
            "Replace the current conversation and draft with the selected session?",
            parent=self.root,
        ):
            return

        self._loading_state = True
        try:
            self.base_model_var.set(config.base_model)
            self.model_path_var.set(config.model_path or "")
            self.base_url_var.set(config.base_url or "")
            self.renderer_var.set(config.renderer_name or "")
            self.max_tokens_var.set(str(config.max_tokens))
            self.temperature_var.set(str(config.temperature))
            self.top_p_var.set(str(config.top_p))
            self.top_k_var.set(str(config.top_k))
            self.seed_var.set("" if config.seed is None else str(config.seed))
            self._set_text_widget_value(self.system_prompt_text, config.system_prompt)
            self._set_text_widget_value(self.input_text, "")
        finally:
            self._loading_state = False

        self.session = session
        self.last_result = None
        self._refresh_transcript()
        self._update_renderer_hint()
        self._update_sampling_preset_indicator()
        self._update_system_prompt_preset_indicator()
        self._update_session_summaries()
        self._update_draft_status()
        self._sync_button_states()
        self.status_var.set(f"Loaded session from {file_path}")
        self._schedule_state_save()

    def reset_system_prompt(self) -> None:
        self._set_text_widget_value(self.system_prompt_text, DEFAULT_SYSTEM_PROMPT)
        self._update_system_prompt_preset_indicator()
        self.status_var.set("System prompt reset to the default preset.")
        self._schedule_state_save()

    def _apply_sampling_preset(self, _event: tk.Event) -> None:
        preset_name = self.sampling_preset_var.get()
        preset = SAMPLING_PRESETS.get(preset_name)
        if preset is None:
            return
        self.temperature_var.set(preset["temperature"])
        self.top_p_var.set(preset["top_p"])
        self.top_k_var.set(preset["top_k"])
        self.status_var.set(f"Applied the {preset_name.lower()} sampling preset.")

    def _apply_system_prompt_preset(self, _event: tk.Event) -> None:
        preset_name = self.system_prompt_preset_var.get()
        prompt = SYSTEM_PROMPT_PRESETS.get(preset_name)
        if prompt is None:
            return
        self._set_text_widget_value(self.system_prompt_text, prompt)
        self.status_var.set(f"Applied the {preset_name.lower()} system prompt preset.")
        self._schedule_state_save()

    def _on_close(self) -> None:
        if self._save_state_after_id is not None:
            self.root.after_cancel(self._save_state_after_id)
            self._save_state_after_id = None
        self._save_state()
        self.root.destroy()


def parse_args(argv: list[str] | None = None) -> LaunchOptions:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    cli_overrides = _collect_cli_overrides(raw_argv)

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
        default=int(os.environ.get("TINKER_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
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
        "--top-k",
        type=int,
        default=int(os.environ.get("TINKER_TOP_K", "-1")),
        help="Top-k sampling. Use -1 to disable.",
    )
    parser.add_argument(
        "--seed",
        default=os.environ.get("TINKER_SEED"),
        help="Optional sampling seed for reproducible generations.",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("TINKER_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="Initial system prompt shown in the GUI.",
    )
    parser.add_argument(
        "--no-restore-state",
        action="store_true",
        help="Start fresh instead of restoring the last locally-saved draft and session.",
    )

    args = parser.parse_args(raw_argv)
    seed = None if args.seed in (None, "") else int(args.seed)

    defaults = ChatConfig(
        base_model=args.base_model,
        model_path=args.model_path,
        base_url=args.base_url,
        renderer_name=args.renderer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=seed,
        system_prompt=args.system_prompt,
    )
    return LaunchOptions(
        defaults=defaults,
        cli_overrides=cli_overrides,
        restore_state=not args.no_restore_state,
    )


def main() -> None:
    launch_options = parse_args()
    root = tk.Tk()
    app = TinkerChatGUI(root, launch_options)
    root.after(50, app.input_text.focus_set)
    root.mainloop()


if __name__ == "__main__":
    main()
