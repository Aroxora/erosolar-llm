"""Built-in tools. All side-effecting tools are constrained to a workspace dir
or carry explicit caveats. Network and code-exec tools are clearly bounded."""

from __future__ import annotations

import ast
import base64
import ipaddress
import json
import operator
import re
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .. import secrets
from .tools import ToolRegistry
from .types import ToolResult

# --- finish (terminal) -----------------------------------------------------

def _finish(answer: str) -> ToolResult:
    return ToolResult(ok=True, content=answer)


# --- calculator (safe arithmetic only) -------------------------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")  # guard against e.g. 9**9**9 DoS
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only arithmetic on numbers is allowed")


def _calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return str(_safe_eval(tree))


# --- web fetch -------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _host_is_blocked(host: str) -> bool:
    """Block SSRF to private/loopback/link-local/reserved ranges (incl. the
    169.254.169.254 cloud-metadata endpoint). Unresolvable -> blocked."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            return True
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return True
    return False


def _http_get(url: str, max_chars: int = 4000) -> ToolResult:
    if not url.startswith(("http://", "https://")):
        return ToolResult(ok=False, error="url must start with http:// or https://")
    host = urlparse(url).hostname
    if not host or _host_is_blocked(host):
        return ToolResult(ok=False, error="blocked host (private/loopback/link-local "
                                          "addresses are not allowed)")
    req = urllib.request.Request(url, headers={"User-Agent": "erosolar-agent/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(2_000_000).decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return ToolResult(ok=False, error=f"fetch failed: {e}")
    text = _WS_RE.sub(" ", _TAG_RE.sub(" ", raw)).strip()
    return ToolResult(ok=True, content=text[: int(max_chars)])


# --- sandboxed file I/O ----------------------------------------------------

def _resolve(workspace: Path, path: str) -> Path:
    p = (workspace / path).resolve()
    if workspace.resolve() not in p.parents and p != workspace.resolve():
        raise ValueError("path escapes the workspace")
    return p


def _make_file_tools(workspace: Path):
    workspace.mkdir(parents=True, exist_ok=True)

    def read_file(path: str, max_chars: int = 8000) -> ToolResult:
        p = _resolve(workspace, path)
        if not p.exists():
            return ToolResult(ok=False, error=f"no such file: {path}")
        return ToolResult(ok=True, content=p.read_text(encoding="utf-8", errors="replace")[: int(max_chars)])

    def write_file(path: str, content: str) -> ToolResult:
        p = _resolve(workspace, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult(ok=True, content=f"wrote {len(content)} chars to {path}")

    def list_dir(path: str = ".") -> ToolResult:
        p = _resolve(workspace, path)
        if not p.exists():
            return ToolResult(ok=False, error=f"no such dir: {path}")
        entries = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
        return ToolResult(ok=True, content="\n".join(entries) or "(empty)")

    return read_file, write_file, list_dir


def _make_python_tool(workspace: Path):
    workspace.mkdir(parents=True, exist_ok=True)

    def run_python(code: str, timeout_s: int = 30) -> ToolResult:
        """Execute Python in a fresh subprocess, cwd=workspace. NOT a hard sandbox —
        enable only for trusted tasks (cfg.enable_python_tool)."""
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True, text=True, timeout=int(timeout_s),
                cwd=str(workspace),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"timed out after {timeout_s}s")
        out = (proc.stdout or "")[-6000:]
        err = (proc.stderr or "")[-2000:]
        if proc.returncode != 0:
            return ToolResult(ok=False, error=f"exit {proc.returncode}\n{err}")
        return ToolResult(ok=True, content=out + (f"\n[stderr]\n{err}" if err else ""))

    return run_python


# --- image/video generation (uses the stored API key) ----------------------

def _image_video_generate(prompt: str, kind: str = "image") -> ToolResult:
    """Generate an image or video from a text prompt via Stability AI's SDXL API.

    IMAGE_VIDEO_GEN_API_KEY  -> Stability AI API key (Bearer token)
    IMAGE_VIDEO_GEN_BASE_URL -> optional custom base URL (defaults to Stability AI)

    Images are saved to a temp dir and the file path + seed are returned.
    Video requires first generating an image internally, then running img2vid."""
    key = secrets.image_video_gen_api_key()
    base = secrets.image_video_gen_base_url(default="https://api.stability.ai")
    if not key:
        return ToolResult(ok=False, error="IMAGE_VIDEO_GEN_API_KEY not set")

    out_dir = Path(tempfile.mkdtemp(prefix="erosolar_gen_"))
    kind = kind or "image"

    if kind == "image":
        return _generate_image(key, base, prompt, out_dir)
    if kind == "video":
        return _generate_video(key, base, prompt, out_dir)
    return ToolResult(ok=False, error=f"unknown kind: {kind!r} (use 'image' or 'video')")


def _generate_image(key: str, base: str, prompt: str, out_dir: Path) -> ToolResult:
    url = f"{base.rstrip('/')}/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    body = json.dumps({
        "text_prompts": [{"text": prompt, "weight": 1.0}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        return ToolResult(ok=False, error=f"API error {e.code}: {detail}")
    except urllib.error.URLError as e:
        return ToolResult(ok=False, error=f"connection failed: {e}")

    artifacts = data.get("artifacts")
    if not artifacts:
        return ToolResult(ok=False, error=f"no artifacts in response: {json.dumps(data)[:300]}")
    b64 = artifacts[0].get("base64")
    if not b64:
        return ToolResult(ok=False, error="artifact missing base64 data")
    seed = artifacts[0].get("seed", "unknown")
    img_bytes = base64.b64decode(b64)

    path = out_dir / f"sdxl_{seed}.png"
    path.write_bytes(img_bytes)
    return ToolResult(ok=True, content=f"generated image saved to {path} (seed={seed}, {len(img_bytes)} bytes)")


def _generate_video(key: str, base: str, prompt: str, out_dir: Path) -> ToolResult:
    img_result = _generate_image(key, base, prompt, out_dir)
    if not img_result.ok:
        return ToolResult(ok=False, error=f"image step failed: {img_result.error}")

    img_path = Path(img_result.content.split("saved to ")[1].split(" (")[0])
    img_bytes = img_path.read_bytes()

    url = f"{base.rstrip('/')}/v2beta/image-to-video"
    boundary = "erosolar-boundary-42"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="frame.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + img_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="seed"\r\n\r\n0\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="cfg_scale"\r\n\r\n1.8\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="motion_bucket_id"\r\n\r\n127\r\n'
        f"--{boundary}--\r\n"
    ).encode()

    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            vid_bytes = resp.read()
        vid = out_dir / f"sdxl_video_{img_path.stem}.mp4"
        vid.write_bytes(vid_bytes)
        return ToolResult(ok=True, content=f"generated video saved to {vid} ({len(vid_bytes)} bytes)")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        return ToolResult(ok=False, error=f"video API error {e.code}: {detail}")
    except urllib.error.URLError as e:
        return ToolResult(ok=False, error=f"video connection failed: {e}")


# --- web search (Tavily, with graceful quota handling) ---------------------

def _web_search(query: str, max_results: int = 5) -> ToolResult:
    from ..integrations import tavily
    from ..integrations.quota import QuotaExhausted
    try:
        res = tavily.search(query, max_results=max_results)
    except QuotaExhausted as e:
        # Friendly "disabled until reset / top up" message — the model can adapt.
        return ToolResult(ok=False, error=str(e))
    except Exception as e:  # noqa: BLE001
        return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
    lines = []
    if res.get("answer"):
        lines.append(f"Answer: {res['answer']}")
    for it in (res.get("results") or [])[:max_results]:
        lines.append(f"- {it.get('title', '')} ({it.get('url', '')})\n  "
                     f"{(it.get('content') or '')[:300]}")
    return ToolResult(ok=True, content="\n".join(lines) or "(no results)")


# --- registry assembly -----------------------------------------------------

def build_default_registry(cfg) -> ToolRegistry:
    reg = ToolRegistry()
    workspace = Path(cfg.workspace_dir)

    reg.add(
        "finish", "Provide the complete final answer and end the task.",
        {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        _finish, terminal=True,
    )
    reg.add(
        "calculator", "Evaluate a numeric arithmetic expression (e.g. '2*(3+4)**2').",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        _calculator,
    )
    reg.add(
        "http_get", "Fetch a URL and return its text content (HTML stripped, truncated).",
        {"type": "object", "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "default": 4000}}, "required": ["url"]},
        _http_get,
    )

    # Tavily web search — only exposed when a key is configured, so the model
    # isn't offered a tool that can't work. Out-of-quota returns a clear message.
    if secrets.get_secret("TAVILY_API_KEY"):
        reg.add(
            "web_search",
            "Search the web for current information (Tavily). Returns top results + a synthesized answer.",
            {"type": "object", "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5}}, "required": ["query"]},
            _web_search,
        )

    read_file, write_file, list_dir = _make_file_tools(workspace)
    reg.add("read_file", "Read a text file from the workspace.",
            {"type": "object", "properties": {"path": {"type": "string"},
             "max_chars": {"type": "integer", "default": 8000}}, "required": ["path"]}, read_file)
    reg.add("write_file", "Write/overwrite a text file in the workspace.",
            {"type": "object", "properties": {"path": {"type": "string"},
             "content": {"type": "string"}}, "required": ["path", "content"]}, write_file)
    reg.add("list_dir", "List entries of a workspace directory.",
            {"type": "object", "properties": {"path": {"type": "string", "default": "."}}}, list_dir)

    reg.add("image_video_generate",
            "Generate an image or video from a text prompt (kind: 'image' or 'video').",
            {"type": "object", "properties": {"prompt": {"type": "string"},
             "kind": {"type": "string", "enum": ["image", "video"], "default": "image"}},
             "required": ["prompt"]}, _image_video_generate)

    if cfg.enable_python_tool:
        reg.add("run_python", "Run Python code in a subprocess (cwd=workspace). Returns stdout.",
                {"type": "object", "properties": {"code": {"type": "string"},
                 "timeout_s": {"type": "integer", "default": 30}}, "required": ["code"]},
                _make_python_tool(workspace))

    return reg
