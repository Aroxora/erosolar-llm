"""Runtime configuration for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import secrets


@dataclass
class AgentConfig:
    # model endpoint (OpenAI-compatible; e.g. a local vLLM server)
    model: str = "erosolar-qwen3-32b"
    base_url: str = ""      # default resolved in __post_init__
    api_key: str = ""

    # sampling
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 1024

    # loop control
    max_steps: int = 24
    reflect_every: int = 6          # inject a self-critique every N steps (0 = off)
    enable_planning: bool = True

    # long-horizon context management
    context_token_budget: int = 24000   # compact transcript above this (rough est.)
    keep_recent_messages: int = 12       # always keep this many recent turns verbatim

    # tools
    tool_allowlist: list = field(default_factory=list)  # empty -> all registered
    enable_python_tool: bool = False     # off by default (executes code in a subprocess)
    workspace_dir: str = ".agent_workspace"

    verbose: bool = True

    def __post_init__(self):
        if not self.base_url:
            self.base_url = secrets.get_secret("OPENAI_BASE_URL", "http://localhost:8000/v1")
        if not self.api_key:
            # vLLM accepts any non-empty key by default
            self.api_key = secrets.get_secret("OPENAI_API_KEY", "EMPTY")
