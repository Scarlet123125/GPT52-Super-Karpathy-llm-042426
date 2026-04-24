import os
import re
import io
import json
import time
import zipfile
import hashlib
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Generator

import streamlit as st
import yaml
import pandas as pd
import altair as alt

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# ----------------------------
# Constants / Config
# ----------------------------

APP_TITLE = "Knowledge Agent WOW v4.1"
DEFAULT_OUTPUT_LANGUAGE = "Traditional Chinese"
OUTPUT_LANGUAGES = ["Traditional Chinese", "English"]

DEFAULT_MODEL = "gemini-3-flash-preview"

MODEL_OPTIONS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "grok-4-fast-reasoning",
    "grok-3-mini",
    # Anthropic models are variable; allow custom entry in UI too.
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
]

PAINTER_STYLES = [
    "Vincent van Gogh",
    "Claude Monet",
    "Pablo Picasso",
    "Salvador Dalí",
    "Rembrandt",
    "Johannes Vermeer",
    "Katsushika Hokusai",
    "Utagawa Hiroshige",
    "Jackson Pollock",
    "Gustav Klimt",
    "Frida Kahlo",
    "Edward Hopper",
    "Henri Matisse",
    "René Magritte",
    "Andy Warhol",
    "Georgia O’Keeffe",
    "Paul Cézanne",
    "Caravaggio",
    "J.M.W. Turner",
    "Wassily Kandinsky",
]

WOW_SUMMARY_MAGICS = [
    "Truth-Table & Claims Audit",
    "Cross-Document Contradiction Finder",
    "Obsidian Linksmith (Graph Booster)",
    "Executive Reframe (Audience Switch)",
    "Study Kit Generator",
    "Action Plan & Checklist",
]

NOTE_MAGICS = [
    "AI Formatting",
    "AI Keywords Highlighter",
    "AI Outline-to-Note",
    "AI Dedup & Merge",
    "AI Citation Helper",
    "AI Obsidian Linking",
]

VISUALIZATION_OPTIONS = [
    "Run Timeline (Gantt-like)",
    "Source Contribution Heatmap (Heuristic)",
    "Topic Map (Top Terms)",
    "Entity Network (Heuristic Co-occurrence)",
    "Risk & Uncertainty Radar (Heuristic)",
    "Version Diff Viewer (Text)",
]

ENV_KEY_NAMES = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "grok": ["XAI_API_KEY", "GROK_API_KEY"],
}

# Workspace base (HF Spaces typically allows /tmp)
WORKSPACE_BASE = os.environ.get("KA_WORKSPACE", "/tmp/knowledge_agent_wow")


# ----------------------------
# Utility: Session State Init
# ----------------------------

def _init_state():
    ss = st.session_state
    ss.setdefault("run_id", None)
    ss.setdefault("run_meta", {})
    ss.setdefault("logs", [])
    ss.setdefault("stage", "Idle")
    ss.setdefault("queue", [])  # list[dict] with file entries
    ss.setdefault("converted", {})  # file_id -> markdown text
    ss.setdefault("converted_paths", {})  # file_id -> path
    ss.setdefault("summary_versions", [])  # list of dict: {id, name, text, created_at, meta}
    ss.setdefault("active_summary_id", None)
    ss.setdefault("magic_outputs", [])  # list of dict outputs
    ss.setdefault("note_versions", [])
    ss.setdefault("agents_yaml_text", "")
    ss.setdefault("agents_yaml_standard_report", "")
    ss.setdefault("keys_user", {})  # provider -> key if user entered
    ss.setdefault("model_global", DEFAULT_MODEL)
    ss.setdefault("model_by_feature", {
        "summary": DEFAULT_MODEL,
        "summary_chat": DEFAULT_MODEL,
        "summary_magic": DEFAULT_MODEL,
        "note_keeper": DEFAULT_MODEL,
        "agents_standardize": DEFAULT_MODEL,
    })
    ss.setdefault("prompt_templates", default_prompt_templates())
    ss.setdefault("output_language", DEFAULT_OUTPUT_LANGUAGE)
    ss.setdefault("ui_language", "Traditional Chinese")
    ss.setdefault("theme", "Light")
    ss.setdefault("painter_style", "None")
    ss.setdefault("last_run_artifacts_dir", None)
    ss.setdefault("timeline_events", [])  # list of dict: {t, stage, doc_id?, msg}
    ss.setdefault("doc_metrics", {})  # doc_id -> metrics dict
    ss.setdefault("stop_flag", False)


# ----------------------------
# Logging & Telemetry
# ----------------------------

def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str, level: str = "INFO", doc_id: Optional[str] = None, stage: Optional[str] = None):
    entry = {
        "time": now_iso(),
        "level": level,
        "stage": stage or st.session_state.get("stage", "Unknown"),
        "doc_id": doc_id,
        "msg": msg,
    }
    st.session_state.logs.append(entry)
    st.session_state.timeline_events.append({
        "time": time.time(),
        "stage": entry["stage"],
        "doc_id": doc_id or "",
        "level": level,
        "msg": msg,
    })


def set_stage(stage: str):
    st.session_state.stage = stage
    log(f"Stage → {stage}", level="INFO", stage=stage)


def should_stop() -> bool:
    return bool(st.session_state.get("stop_flag", False))


def reset_stop():
    st.session_state.stop_flag = False


# ----------------------------
# Prompt Templates
# ----------------------------

def default_prompt_templates() -> Dict[str, str]:
    # Keep prompts editable in UI.
    return {
        "summary_system": (
            "You are Knowledge Agent WOW. You transform multiple converted Markdown sources into a single, "
            "well-structured integrated Markdown report suitable for Obsidian. "
            "You must preserve meaning, deduplicate overlaps, and keep provenance.\n"
        ),
        "summary_user": (
            "OUTPUT LANGUAGE: {output_language}\n"
            "TASK: Create ONE integrated summary in Markdown that synthesizes ALL sources below.\n"
            "HARD CONSTRAINT: The output MUST be 3000–4000 words (approximate word count).\n"
            "REQUIREMENTS:\n"
            "- Add YAML frontmatter at the top with title, language, sources, run_id.\n"
            "- Use clear headings: Executive Overview, Key Themes (Cross-Document), Document-by-Document Highlights, "
            "Conflicts/Gaps/Uncertainties, Actionable Insights/Next Steps, Appendix: Source Map.\n"
            "- Include source references (file names) in a lightweight way.\n"
            "- Do not include raw logs or tool output.\n\n"
            "SOURCES (each is already cleaned Markdown):\n"
            "{sources_block}\n"
        ),
        "summary_adjust_length": (
            "Your previous integrated summary does not meet the 3000–4000 word constraint.\n"
            "OUTPUT LANGUAGE: {output_language}\n"
            "Please rewrite the summary to strictly fit 3000–4000 words while keeping the same structure, "
            "and without losing critical information.\n\n"
            "CURRENT SUMMARY:\n"
            "{current_summary}\n"
        ),
        "summary_chat": (
            "Use the integrated summary as the only context. Answer the user's request.\n"
            "OUTPUT LANGUAGE: {output_language}\n"
            "If you need to modify the summary, return Markdown that can replace/append to the summary.\n\n"
            "SUMMARY:\n{summary}\n\n"
            "USER:\n{user_prompt}\n"
        ),
        "magic_system": (
            "You are Knowledge Agent WOW. Apply a specific transformation (WOW Magic) to the integrated summary. "
            "Maintain Markdown, preserve factual integrity, and provide actionable structure.\n"
        ),
        "magic_user": (
            "OUTPUT LANGUAGE: {output_language}\n"
            "MAGIC: {magic_name}\n"
            "OPTIONAL STYLE: {style}\n"
            "Apply the magic to the summary below.\n\n"
            "SUMMARY:\n{summary}\n"
        ),
        "note_organize": (
            "OUTPUT LANGUAGE: {output_language}\n"
            "Transform the following note (text or markdown) into a well-organized Markdown note.\n"
            "Add headings, bullet points, and a short TL;DR.\n\n"
            "NOTE:\n{note}\n"
        ),
        "note_magic": (
            "OUTPUT LANGUAGE: {output_language}\n"
            "NOTE MAGIC: {magic_name}\n"
            "If keywords are provided, highlight them using the chosen color style in Markdown/HTML.\n\n"
            "KEYWORDS: {keywords}\n"
            "COLOR: {color}\n\n"
            "NOTE:\n{note}\n"
        ),
        "agents_standardize": (
            "You standardize an agents.yaml configuration.\n"
            "Return ONLY valid YAML.\n"
            "If the input is non-standard, transform it into a standardized schema:\n"
            "agents:\n"
            "  - name: string\n"
            "    description: string\n"
            "    model: string\n"
            "    prompts:\n"
            "      system: string\n"
            "      user: string\n"
            "    enabled_tools: [string]\n"
            "    extensions: object\n\n"
            "INPUT YAML:\n{yaml_text}\n"
        ),
    }


# ----------------------------
# Workspace / Run Management
# ----------------------------

def new_run_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    salt = hashlib.sha1(str(time.time()).encode("utf-8")).hexdigest()[:8]
    return f"run_{ts}_{salt}"


def ensure_dirs(run_id: str) -> Dict[str, str]:
    base = os.path.join(WORKSPACE_BASE, "raw", "imports", run_id)
    paths = {
        "base": base,
        "inputs": os.path.join(base, "inputs"),
        "markdown": os.path.join(base, "markdown"),
        "summary": os.path.join(base, "summary"),
        "logs": os.path.join(base, "logs"),
        "meta": os.path.join(base, "meta"),
        "assets": os.path.join(base, "assets"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    st.session_state.last_run_artifacts_dir = base
    return paths


def write_run_meta(paths: Dict[str, str], meta: dict):
    meta_path = os.path.join(paths["meta"], "run.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_logs(paths: Dict[str, str]):
    log_path = os.path.join(paths["logs"], "run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for e in st.session_state.logs:
            f.write(f"[{e['time']}][{e['level']}][{e['stage']}][{e.get('doc_id','')}] {e['msg']}\n")


# ----------------------------
# API Key Policy
# ----------------------------

def get_env_key(provider: str) -> Optional[str]:
    for name in ENV_KEY_NAMES.get(provider, []):
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    return None


def get_effective_key(provider: str) -> Optional[str]:
    env_k = get_env_key(provider)
    if env_k:
        return env_k
    user_k = st.session_state.keys_user.get(provider)
    if user_k and user_k.strip():
        return user_k.strip()
    return None


def model_provider(model_name: str) -> str:
    m = (model_name or "").lower()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("gpt-"):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("grok"):
        return "grok"
    # Default heuristic: treat unknown as openai-compatible
    return "openai"


# ----------------------------
# LLM Invocation (Streaming-first, fallback to non-stream)
# ----------------------------

@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    usage: dict


def _import_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except Exception:
        return None


def _import_anthropic():
    try:
        import anthropic
        return anthropic
    except Exception:
        return None


def _import_google_genai():
    # google-genai (newer) import style
    try:
        from google import genai
        return genai
    except Exception:
        return None


def llm_stream(system_prompt: str, user_prompt: str, model: str) -> Generator[str, None, LLMResult]:
    """
    Yields text chunks for Streamlit rendering.
    Returns final LLMResult via generator return (StopIteration.value).
    """
    provider = model_provider(model)
    key = get_effective_key(provider)
    if not key:
        raise RuntimeError(f"Missing API key for provider: {provider}")

    usage = {}
    collected = []

    # ---------------- GEMINI ----------------
    if provider == "gemini":
        genai = _import_google_genai()
        if genai is None:
            raise RuntimeError("google-genai is not installed. Add `google-genai` to requirements.txt.")

        client = genai.Client(api_key=key)

        # Use a simple text call; streaming supported by generate_content_stream
        contents = [
            {"role": "system", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": user_prompt}]},
        ]

        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
            )
            for ev in stream:
                if should_stop():
                    break
                chunk = getattr(ev, "text", None) or ""
                if chunk:
                    collected.append(chunk)
                    yield chunk
        except Exception as e:
            raise RuntimeError(f"Gemini streaming error: {e}") from e

        text = "".join(collected).strip()
        return LLMResult(text=text, provider=provider, model=model, usage=usage)

    # ---------------- OPENAI / GROK (OpenAI-compatible for Grok) ----------------
    if provider in ("openai", "grok"):
        OpenAI = _import_openai()
        if OpenAI is None:
            raise RuntimeError("openai is not installed. Add `openai` to requirements.txt.")

        # Grok uses xAI OpenAI-compatible base URL
        if provider == "grok":
            client = OpenAI(api_key=key, base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"))
        else:
            client = OpenAI(api_key=key)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            for event in resp:
                if should_stop():
                    break
                delta = ""
                try:
                    delta = event.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta:
                    collected.append(delta)
                    yield delta
        except Exception as e:
            raise RuntimeError(f"{provider} streaming error: {e}") from e

        text = "".join(collected).strip()
        return LLMResult(text=text, provider=provider, model=model, usage=usage)

    # ---------------- ANTHROPIC ----------------
    if provider == "anthropic":
        anthropic = _import_anthropic()
        if anthropic is None:
            raise RuntimeError("anthropic is not installed. Add `anthropic` to requirements.txt.")

        client = anthropic.Anthropic(api_key=key)

        try:
            with client.messages.stream(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=4096,
            ) as stream:
                for text in stream.text_stream:
                    if should_stop():
                        break
                    if text:
                        collected.append(text)
                        yield text
        except Exception as e:
            raise RuntimeError(f"Anthropic streaming error: {e}") from e

        text = "".join(collected).strip()
        return LLMResult(text=text, provider=provider, model=model, usage=usage)

    raise RuntimeError(f"Unknown provider for model: {model}")


def run_llm(system_prompt: str, user_prompt: str, model: str, stream_to_ui: bool = True) -> LLMResult:
    """
    Wrapper that streams to UI if requested, otherwise collects internally.
    """
    gen = llm_stream(system_prompt, user_prompt, model)
    collected = []
    if stream_to_ui:
        placeholder = st.empty()
        buf = ""

        try:
            for chunk in gen:
                buf += chunk
                # Keep UI responsive but not too slow
                placeholder.markdown(buf)
                collected.append(chunk)
        except Exception:
            # ensure exception is visible
            raise
        try:
            result = gen.send(None)  # finalize
            return result
        except StopIteration as si:
            return si.value
    else:
        try:
            for chunk in gen:
                collected.append(chunk)
        except Exception:
            raise
        try:
            result = gen.send(None)
            return result
        except StopIteration as si:
            return si.value


# ----------------------------
# Document Handling / Conversion (skill.md-inspired)
# ----------------------------

def slugify(name: str) -> str:
    name = re.sub(r"[^\w\s\-\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.replace(" ", "_")
    return name[:120] if name else "untitled"


def pangu_spacing(text: str) -> str:
    # Minimal CJK/ASCII spacing heuristic (lightweight)
    # Insert space between CJK and ASCII when adjacent.
    def is_cjk(ch):
        return (
            "\u4e00" <= ch <= "\u9fff"
            or "\u3400" <= ch <= "\u4dbf"
            or "\u3040" <= ch <= "\u30ff"
        )

    out = []
    for i, ch in enumerate(text):
        if i > 0:
            prev = text[i - 1]
            if (is_cjk(prev) and re.match(r"[A-Za-z0-9]", ch)) or (re.match(r"[A-Za-z0-9]", prev) and is_cjk(ch)):
                if out and out[-1] != " ":
                    out.append(" ")
        out.append(ch)
    return "".join(out)


def cleanup_common(md: str) -> str:
    md = md.replace("------", "——")
    # remove pandoc attributes patterns
    md = re.sub(r"\{\.[^}]+\}", "", md)
    md = re.sub(r"\[\]\{#[^\}]+\}", "", md)
    md = re.sub(r":::.*?\n", "", md)
    md = re.sub(r"\.immersive-translate-[\w\-]+", "", md)
    md = re.sub(r"\.notranslate", "", md)
    # compress blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = pangu_spacing(md)
    return md.strip() + "\n"


def detect_title_from_md(md: str, fallback: str) -> str:
    m = re.search(r"^\s*#\s+(.+?)\s*$", md, flags=re.M)
    if m:
        return m.group(1).strip()
    return fallback


def build_frontmatter(title: str, source_filename: str, doc_type: str, origin: str = "external") -> str:
    fm = {
        "title": title,
        "origin": origin,
        "type": doc_type,
        "source_filename": source_filename,
        "created_at": now_iso(),
    }
    y = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{y}\n---\n\n"


def convert_txt_to_md(text: str, filename: str) -> Tuple[str, dict]:
    raw = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Heuristic headings: lines that look like "1. Title" or "CHAPTER X" or all-caps > 6 chars
    lines = raw.split("\n")
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if re.match(r"^\d+(\.\d+)*\s+.+", s) and len(s) < 120:
            out.append("## " + s)
        elif re.match(r"^(chapter|section)\s+\d+", s.lower()) and len(s) < 120:
            out.append("## " + s)
        elif (s.isupper() and len(s) >= 8 and len(s) < 80 and re.search(r"[A-Z]", s)):
            out.append("## " + s.title())
        else:
            out.append(s)
    md_body = "\n".join(out)
    md_body = cleanup_common(md_body)
    title = detect_title_from_md(md_body, os.path.splitext(filename)[0])
    md = build_frontmatter(title=title, source_filename=filename, doc_type="document") + md_body
    metrics = {
        "title": title,
        "chars": len(md),
        "words_est": len(re.findall(r"\S+", md_body)),
        "format": "txt",
    }
    return md, metrics


def convert_md_to_md(md: str, filename: str) -> Tuple[str, dict]:
    body = md.replace("\r\n", "\n").replace("\r", "\n").strip()
    # If user already has frontmatter, keep it, but ensure required keys exist.
    # For simplicity, we preserve content; we also prepend a standardized frontmatter if none exists.
    has_fm = body.startswith("---\n") and ("\n---\n" in body[4:])
    if not has_fm:
        title = detect_title_from_md(body, os.path.splitext(filename)[0])
        body = cleanup_common(body)
        out = build_frontmatter(title=title, source_filename=filename, doc_type="document") + body
    else:
        # still cleanup body lightly (but avoid breaking YAML by not touching top block)
        parts = body.split("\n---\n", 1)
        fm = parts[0] + "\n---\n"
        rest = parts[1] if len(parts) > 1 else ""
        rest = cleanup_common(rest)
        title = detect_title_from_md(rest, os.path.splitext(filename)[0])
        out = fm + "\n" + rest
    metrics = {
        "title": detect_title_from_md(out, os.path.splitext(filename)[0]),
        "chars": len(out),
        "words_est": len(re.findall(r"\S+", out)),
        "format": "md",
    }
    return out, metrics


def extract_pdf_text_pymupdf(pdf_bytes: bytes, filename: str) -> Tuple[str, dict]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Add `pymupdf` to requirements.txt.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip() or os.path.splitext(filename)[0]

    # TOC-based headings if available
    toc = doc.get_toc()  # list of [level, title, page]
    toc_map = {}  # page_index -> list of (level, title)
    for level, t, page in toc:
        idx = max(0, int(page) - 1)
        toc_map.setdefault(idx, []).append((level, t))

    page_texts = []
    header_candidates = {}
    footer_candidates = {}

    for i in range(doc.page_count):
        if should_stop():
            break

        page = doc.load_page(i)
        text = page.get_text("text") or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        # Capture potential headers/footers (first/last non-empty lines)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if lines:
            header_candidates[lines[0]] = header_candidates.get(lines[0], 0) + 1
            footer_candidates[lines[-1]] = footer_candidates.get(lines[-1], 0) + 1

        page_texts.append(lines)

    # Heuristic: remove frequent header/footer lines
    header_to_remove = {k for k, v in header_candidates.items() if v >= max(3, doc.page_count // 3)}
    footer_to_remove = {k for k, v in footer_candidates.items() if v >= max(3, doc.page_count // 3)}

    out_lines = []
    removed_hf = 0
    removed_pageno = 0

    for i, lines in enumerate(page_texts):
        if should_stop():
            break

        # Insert TOC headings at page start
        if i in toc_map:
            for level, t in toc_map[i]:
                if not t.strip():
                    continue
                hashes = "##" if level == 1 else "###" if level == 2 else "####"
                out_lines.append(f"{hashes} {t.strip()}")
                out_lines.append("")

        for ln in lines:
            if ln in header_to_remove or ln in footer_to_remove:
                removed_hf += 1
                continue
            # remove standalone page number lines
            if re.match(r"^\d{1,4}$", ln):
                removed_pageno += 1
                continue
            # remove PUA noise lines
            if re.search(r"[\ue000-\uf8ff]", ln):
                removed_pageno += 1
                continue
            out_lines.append(ln)
        out_lines.append("")

    md_body = "\n".join(out_lines)
    md_body = cleanup_common(md_body)

    md = build_frontmatter(title=title, source_filename=filename, doc_type="report") + md_body
    metrics = {
        "title": title,
        "chars": len(md),
        "words_est": len(re.findall(r"\S+", md_body)),
        "format": "pdf",
        "pages": doc.page_count,
        "removed_headers_footers": removed_hf,
        "removed_page_numbers": removed_pageno,
        "toc_entries": len(toc),
    }
    doc.close()
    return md, metrics


def convert_document(file_entry: dict, paths: Dict[str, str]) -> Tuple[str, dict, str]:
    """
    Returns: (markdown_text, metrics, output_path)
    """
    fid = file_entry["id"]
    filename = file_entry["name"]
    ext = file_entry["ext"].lower()

    set_stage("Converting")
    log(f"Converting {filename}", doc_id=fid, stage="Converting")

    if ext == ".pdf":
        md, metrics = extract_pdf_text_pymupdf(file_entry["bytes"], filename)
    elif ext == ".txt":
        md, metrics = convert_txt_to_md(file_entry["text"], filename)
    elif ext == ".md":
        md, metrics = convert_md_to_md(file_entry["text"], filename)
    else:
        raise RuntimeError(f"Unsupported file type: {ext}")

    # Write markdown file
    title_slug = slugify(metrics.get("title") or os.path.splitext(filename)[0])
    out_name = f"{title_slug}.md"
    out_path = os.path.join(paths["markdown"], out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    log(f"Wrote Markdown → {out_name}", doc_id=fid, stage="Converting")
    return md, metrics, out_path


# ----------------------------
# Summary Assembly
# ----------------------------

def build_sources_block(converted: Dict[str, str], queue: List[dict]) -> str:
    parts = []
    for fe in queue:
        fid = fe["id"]
        if fid not in converted:
            continue
        label = fe["name"]
        parts.append(f"\n\n---\n# SOURCE: {label}\n---\n\n{converted[fid]}\n")
    return "\n".join(parts).strip()


def word_count_approx(text: str) -> int:
    # Works for English + CJK approximations (counts tokens separated by whitespace; CJK may be undercounted).
    # Still useful for enforcement loop.
    return len(re.findall(r"\S+", text))


def ensure_summary_length(summary_md: str, output_language: str, model: str, max_rounds: int = 2) -> str:
    wc = word_count_approx(summary_md)
    if 3000 <= wc <= 4000:
        return summary_md

    log(f"Summary word count ~{wc} (needs 3000–4000). Attempting adjustment.", level="WARN", stage="Summarizing")

    system_p = st.session_state.prompt_templates["summary_system"]
    for _ in range(max_rounds):
        if should_stop():
            break
        user_p = st.session_state.prompt_templates["summary_adjust_length"].format(
            output_language=output_language,
            current_summary=summary_md,
        )
        result = run_llm(system_p, user_p, model=model, stream_to_ui=True)
        summary_md = result.text.strip()
        wc = word_count_approx(summary_md)
        log(f"Adjusted summary word count ~{wc}", stage="Summarizing")
        if 3000 <= wc <= 4000:
            return summary_md

    return summary_md


def save_summary(paths: Dict[str, str], summary_md: str) -> str:
    out_path = os.path.join(paths["summary"], "summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    return out_path


def add_summary_version(name: str, text: str, meta: dict):
    vid = hashlib.sha1((name + str(time.time())).encode("utf-8")).hexdigest()[:10]
    st.session_state.summary_versions.append({
        "id": vid,
        "name": name,
        "text": text,
        "created_at": now_iso(),
        "meta": meta,
    })
    st.session_state.active_summary_id = vid


def get_active_summary_text() -> str:
    sid = st.session_state.active_summary_id
    for v in st.session_state.summary_versions[::-1]:
        if v["id"] == sid:
            return v["text"]
    # fallback last
    if st.session_state.summary_versions:
        return st.session_state.summary_versions[-1]["text"]
    return ""


# ----------------------------
# Magics (Summary)
# ----------------------------

def run_summary_magic(magic_name: str, style: str):
    summary = get_active_summary_text().strip()
    if not summary:
        st.error("No active summary found. Generate a summary first.")
        return

    set_stage("Magic")
    log(f"Running magic: {magic_name}", stage="Magic")

    system_p = st.session_state.prompt_templates["magic_system"]
    user_p = st.session_state.prompt_templates["magic_user"].format(
        output_language=st.session_state.output_language,
        magic_name=magic_name,
        style=style or "None",
        summary=summary,
    )

    model = st.session_state.model_by_feature.get("summary_magic", st.session_state.model_global)
    result = run_llm(system_p, user_p, model=model, stream_to_ui=True)
    out = result.text.strip()

    add_summary_version(
        name=f"Magic: {magic_name}",
        text=out,
        meta={"magic": magic_name, "style": style, "model": model, "provider": result.provider},
    )
    st.session_state.magic_outputs.append({
        "time": now_iso(),
        "magic": magic_name,
        "style": style,
        "model": model,
        "text": out,
    })
    log(f"Magic completed: {magic_name}", level="SUCCESS", stage="Magic")


# ----------------------------
# Note Keeper
# ----------------------------

def add_note_version(name: str, text: str, meta: dict):
    vid = hashlib.sha1((name + str(time.time())).encode("utf-8")).hexdigest()[:10]
    st.session_state.note_versions.append({
        "id": vid,
        "name": name,
        "text": text,
        "created_at": now_iso(),
        "meta": meta,
    })


def run_note_organize(note_text: str):
    if not note_text.strip():
        st.error("Please paste a note first.")
        return
    set_stage("NoteKeeper")
    log("Organizing note", stage="NoteKeeper")
    model = st.session_state.model_by_feature.get("note_keeper", st.session_state.model_global)
    system_p = "You are a precise note organizer."
    user_p = st.session_state.prompt_templates["note_organize"].format(
        output_language=st.session_state.output_language,
        note=note_text,
    )
    result = run_llm(system_p, user_p, model=model, stream_to_ui=True)
    out = result.text.strip()
    add_note_version("Organized Note", out, {"model": model, "provider": result.provider})
    log("Note organized", level="SUCCESS", stage="NoteKeeper")


def run_note_magic(note_text: str, magic_name: str, keywords: str = "", color: str = "yellow"):
    if not note_text.strip():
        st.error("Please paste or select a note first.")
        return
    set_stage("NoteKeeper")
    log(f"Running note magic: {magic_name}", stage="NoteKeeper")
    model = st.session_state.model_by_feature.get("note_keeper", st.session_state.model_global)
    system_p = "You are an Obsidian-focused note transformer."
    user_p = st.session_state.prompt_templates["note_magic"].format(
        output_language=st.session_state.output_language,
        magic_name=magic_name,
        keywords=keywords or "",
        color=color or "yellow",
        note=note_text,
    )
    result = run_llm(system_p, user_p, model=model, stream_to_ui=True)
    out = result.text.strip()
    add_note_version(f"Note Magic: {magic_name}", out, {"magic": magic_name, "model": model, "provider": result.provider})
    log(f"Note magic completed: {magic_name}", level="SUCCESS", stage="NoteKeeper")


# ----------------------------
# agents.yaml standardization
# ----------------------------

def deterministic_standardize_agents_yaml(yaml_text: str) -> Tuple[str, str]:
    """
    Attempts a deterministic standardization without LLM.
    If parsing fails, return original and a report.
    """
    report = []
    try:
        data = yaml.safe_load(yaml_text) if yaml_text.strip() else None
    except Exception as e:
        return yaml_text, f"YAML parse error; cannot deterministic-standardize: {e}"

    if data is None:
        std = {"agents": []}
        return yaml.safe_dump(std, allow_unicode=True, sort_keys=False), "Empty YAML → standardized to {agents: []}"

    # If already has 'agents' list with required keys, keep but fill defaults
    agents = None
    if isinstance(data, dict) and "agents" in data and isinstance(data["agents"], list):
        agents = data["agents"]
        report.append("Detected top-level 'agents' list.")
    elif isinstance(data, list):
        agents = data
        report.append("Detected YAML as a list → treating as agents list.")
    elif isinstance(data, dict):
        # attempt to find plausible agent list
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                agents = v
                report.append(f"Guessed agents list from key '{k}'.")
                break

    if agents is None:
        # Wrap unknown structure
        std = {
            "agents": [{
                "name": "default",
                "description": "Auto-wrapped agent",
                "model": st.session_state.model_global,
                "prompts": {"system": "", "user": ""},
                "enabled_tools": [],
                "extensions": data,
            }]
        }
        report.append("Could not infer agent structure → wrapped into default agent with extensions.")
        return yaml.safe_dump(std, allow_unicode=True, sort_keys=False), "\n".join(report)

    std_agents = []
    for i, a in enumerate(agents):
        if not isinstance(a, dict):
            std_agents.append({
                "name": f"agent_{i+1}",
                "description": "Auto-converted from non-dict",
                "model": st.session_state.model_global,
                "prompts": {"system": "", "user": str(a)},
                "enabled_tools": [],
                "extensions": {},
            })
            report.append(f"Agent {i+1}: non-dict converted.")
            continue

        name = a.get("name") or a.get("id") or a.get("agent") or f"agent_{i+1}"
        desc = a.get("description") or a.get("desc") or ""
        model = a.get("model") or a.get("llm") or st.session_state.model_global

        prompts = a.get("prompts")
        if not isinstance(prompts, dict):
            prompts = {"system": a.get("system", ""), "user": a.get("prompt", a.get("user", ""))}
        prompts.setdefault("system", "")
        prompts.setdefault("user", "")

        tools = a.get("enabled_tools") or a.get("tools") or []
        if not isinstance(tools, list):
            tools = [str(tools)]

        extensions = {k: v for k, v in a.items() if k not in {"name", "id", "agent", "description", "desc", "model", "llm", "prompts", "system", "user", "prompt", "enabled_tools", "tools"}}

        std_agents.append({
            "name": str(name),
            "description": str(desc),
            "model": str(model),
            "prompts": {"system": str(prompts.get("system", "")), "user": str(prompts.get("user", ""))},
            "enabled_tools": [str(x) for x in tools],
            "extensions": extensions,
        })

    std = {"agents": std_agents}
    return yaml.safe_dump(std, allow_unicode=True, sort_keys=False), "\n".join(report) or "Standardized successfully."


def llm_standardize_agents_yaml(yaml_text: str) -> Tuple[str, str]:
    """
    Uses LLM to standardize; returns (yaml, report).
    """
    set_stage("agents.yaml")
    log("Standardizing agents.yaml with LLM", stage="agents.yaml")
    model = st.session_state.model_by_feature.get("agents_standardize", st.session_state.model_global)
    system_p = "You are a YAML configuration expert. Return only YAML."
    user_p = st.session_state.prompt_templates["agents_standardize"].format(yaml_text=yaml_text)
    result = run_llm(system_p, user_p, model=model, stream_to_ui=True)
    out = result.text.strip()
    # Basic validation
    try:
        _ = yaml.safe_load(out)
        report = f"LLM-standardized successfully (model={model}, provider={result.provider})."
    except Exception as e:
        report = f"LLM output was not valid YAML: {e}. Returning raw output for manual fix."
    return out, report


# ----------------------------
# Dashboard Visualizations (lightweight heuristics)
# ----------------------------

def _df_timeline() -> pd.DataFrame:
    ev = st.session_state.timeline_events
    if not ev:
        return pd.DataFrame(columns=["t", "stage", "doc_id", "level", "msg"])
    t0 = min(e["time"] for e in ev)
    rows = []
    for e in ev:
        rows.append({
            "t": e["time"] - t0,
            "stage": e["stage"],
            "doc_id": e["doc_id"] or "",
            "level": e["level"],
            "msg": e["msg"],
        })
    return pd.DataFrame(rows)


def viz_run_timeline():
    df = _df_timeline()
    if df.empty:
        st.info("No timeline events yet.")
        return
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X("t:Q", title="Time (s since start)"),
        y=alt.Y("stage:N", title="Stage"),
        color=alt.Color("level:N"),
        tooltip=["t", "stage", "doc_id", "level", "msg"],
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df.tail(50), use_container_width=True)


def viz_source_contribution_heatmap():
    # Heuristic: measure overlap of top terms from each source with each summary section.
    summary = get_active_summary_text()
    if not summary.strip():
        st.info("Generate a summary first.")
        return
    sources = []
    for fe in st.session_state.queue:
        fid = fe["id"]
        if fid in st.session_state.converted:
            sources.append((fe["name"], st.session_state.converted[fid]))

    if not sources:
        st.info("No converted sources found.")
        return

    sections = re.split(r"\n(?=##\s+)", summary)
    section_titles = []
    section_texts = []
    for sec in sections:
        m = re.match(r"^##\s+(.+)", sec.strip())
        title = m.group(1).strip() if m else "Frontmatter/Intro"
        section_titles.append(title[:50])
        section_texts.append(sec)

    def top_terms(text, k=30):
        words = re.findall(r"[A-Za-z]{3,}|\u4e00-\u9fff", text)
        words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return set(sorted(freq, key=freq.get, reverse=True)[:k])

    src_terms = {name: top_terms(txt, 40) for name, txt in sources}
    rows = []
    for s_title, s_txt in zip(section_titles, section_texts):
        sec_terms = top_terms(s_txt, 40)
        for src_name, _ in sources:
            overlap = len(sec_terms & src_terms[src_name])
            rows.append({"Section": s_title, "Source": src_name, "OverlapScore": overlap})

    df = pd.DataFrame(rows)
    heat = alt.Chart(df).mark_rect().encode(
        x=alt.X("Source:N", sort=None),
        y=alt.Y("Section:N", sort=None),
        color=alt.Color("OverlapScore:Q"),
        tooltip=["Section", "Source", "OverlapScore"],
    ).properties(height=320)
    st.altair_chart(heat, use_container_width=True)


def viz_topic_map():
    summary = get_active_summary_text()
    if not summary.strip():
        st.info("Generate a summary first.")
        return

    # Basic term frequency for English words; CJK topic modeling is out of scope here.
    words = re.findall(r"[A-Za-z]{3,}", summary.lower())
    stop = set(["the", "and", "for", "with", "that", "this", "from", "into", "are", "was", "were", "have", "has"])
    freq = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1

    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:40]
    if not items:
        st.info("Not enough English terms to build a topic map (try English output or mixed content).")
        return
    df = pd.DataFrame(items, columns=["term", "count"])
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("count:Q", title="Frequency"),
        y=alt.Y("term:N", sort="-x", title="Term"),
        tooltip=["term", "count"],
    ).properties(height=520)
    st.altair_chart(chart, use_container_width=True)


def viz_entity_network():
    # Heuristic co-occurrence network from capitalized tokens (English) or bracket links [[...]].
    summary = get_active_summary_text()
    if not summary.strip():
        st.info("Generate a summary first.")
        return

    links = re.findall(r"\[\[([^\]]+)\]\]", summary)
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", summary)
    entities = links + caps
    entities = [e.strip() for e in entities if 2 <= len(e.strip()) <= 40]
    if not entities:
        st.info("No entities detected (try applying Obsidian Linksmith magic first).")
        return

    # Sliding window co-occurrence
    tokens = re.split(r"[\n\.]+", summary)
    edges = {}
    for sent in tokens:
        present = set()
        for e in entities[:300]:  # cap for performance
            if e in sent:
                present.add(e)
        present = list(present)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = sorted([present[i], present[j]])
                edges[(a, b)] = edges.get((a, b), 0) + 1

    top_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)[:80]
    if not top_edges:
        st.info("Not enough co-occurrence edges to visualize.")
        return

    df = pd.DataFrame([{"a": a, "b": b, "w": w} for (a, b), w in top_edges])
    st.dataframe(df, use_container_width=True)
    st.caption("Heuristic network: edges represent co-occurrence within the same sentence/segment.")


def viz_risk_radar():
    summary = get_active_summary_text()
    if not summary.strip():
        st.info("Generate a summary first.")
        return
    # Heuristic risk scoring via keyword hits
    text = summary.lower()
    categories = {
        "Uncertainty": ["uncertain", "unknown", "uncertainty", "未確定", "不確定", "未知"],
        "Conflict": ["contradiction", "conflict", "inconsistent", "矛盾", "衝突", "不一致"],
        "Missing Data": ["missing", "lack", "insufficient", "不足", "缺少", "欠缺"],
        "Bias/Assumption": ["assume", "assumption", "bias", "假設", "偏誤", "偏差"],
        "Operational Risk": ["risk", "failure", "issue", "風險", "失敗", "問題"],
    }
    rows = []
    for cat, kws in categories.items():
        score = 0
        for kw in kws:
            score += text.count(kw.lower())
        rows.append({"category": cat, "score": min(score, 30)})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("score:Q", title="Heuristic Score (capped)"),
        y=alt.Y("category:N", sort="-x"),
        tooltip=["category", "score"],
    ).properties(height=240)
    st.altair_chart(chart, use_container_width=True)
    st.caption("This is a heuristic indicator, not a factual risk assessment.")


def viz_version_diff():
    versions = st.session_state.summary_versions
    if len(versions) < 2:
        st.info("Need at least 2 summary versions to diff (generate summary, then apply a magic).")
        return

    vnames = [f"{v['created_at']} — {v['name']} ({v['id']})" for v in versions]
    a = st.selectbox("Version A", vnames, index=max(0, len(vnames) - 2))
    b = st.selectbox("Version B", vnames, index=len(vnames) - 1)

    ida = a.split("(")[-1].split(")")[0]
    idb = b.split("(")[-1].split(")")[0]

    ta = next(v["text"] for v in versions if v["id"] == ida)
    tb = next(v["text"] for v in versions if v["id"] == idb)

    # Lightweight diff: show changed lines counts + excerpts
    la = ta.splitlines()
    lb = tb.splitlines()

    set_a = set(la)
    set_b = set(lb)
    removed = [x for x in la if x not in set_b]
    added = [x for x in lb if x not in set_a]

    st.write(f"Removed lines: {len(removed)} | Added lines: {len(added)}")
    st.subheader("Added (sample)")
    st.code("\n".join(added[:80]) or "(none)")
    st.subheader("Removed (sample)")
    st.code("\n".join(removed[:80]) or "(none)")


# ----------------------------
# Export Packager
# ----------------------------

def build_export_zip(artifacts_dir: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(artifacts_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, artifacts_dir)
                z.write(full, arcname=rel)
    return buf.getvalue()


# ----------------------------
# UI Components
# ----------------------------

def render_header():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    cols = st.columns([1.2, 1.1, 1.2, 1.5])
    with cols[0]:
        st.session_state.theme = st.selectbox("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
    with cols[1]:
        st.session_state.ui_language = st.selectbox("UI Language", ["Traditional Chinese", "English"], index=0 if st.session_state.ui_language == "Traditional Chinese" else 1)
    with cols[2]:
        st.session_state.output_language = st.selectbox("Output Language", OUTPUT_LANGUAGES, index=0 if st.session_state.output_language == DEFAULT_OUTPUT_LANGUAGE else 1)
    with cols[3]:
        st.session_state.model_global = st.selectbox("Global Model Default", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state.model_global) if st.session_state.model_global in MODEL_OPTIONS else 0)

    # WOW indicator
    st.info(f"WOW Indicator — Stage: **{st.session_state.stage}** | Run: **{st.session_state.run_id or 'N/A'}**")


def render_sidebar_settings():
    with st.sidebar:
        st.header("Settings & Keys")

        # Keys panel respecting env visibility rule
        for provider in ["gemini", "openai", "anthropic", "grok"]:
            env_k = get_env_key(provider)
            if env_k:
                st.success(f"{provider.upper()} key: configured via environment")
            else:
                st.session_state.keys_user[provider] = st.text_input(
                    f"{provider.upper()} API Key",
                    type="password",
                    value=st.session_state.keys_user.get(provider, ""),
                    help="Stored in session only. Not displayed once set.",
                )

        st.divider()
        st.subheader("Feature Models")
        for feat in ["summary", "summary_chat", "summary_magic", "note_keeper", "agents_standardize"]:
            st.session_state.model_by_feature[feat] = st.selectbox(
                f"Model for {feat}",
                MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(st.session_state.model_by_feature.get(feat, st.session_state.model_global))
                if st.session_state.model_by_feature.get(feat, st.session_state.model_global) in MODEL_OPTIONS else 0
            )

        st.divider()
        st.subheader("Painter Style (optional)")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.session_state.painter_style = st.selectbox("Style", ["None"] + PAINTER_STYLES, index=0)
        with c2:
            if st.button("Jackpot"):
                import random
                st.session_state.painter_style = random.choice(PAINTER_STYLES)
                st.rerun()

        st.divider()
        st.subheader("Run Control")
        if st.button("Stop / Cancel"):
            st.session_state.stop_flag = True
            log("User requested stop.", level="WARN")
        if st.button("Reset Stop Flag"):
            reset_stop()
            log("Stop flag reset.", level="INFO")

        st.divider()
        if st.button("Clear Session Outputs"):
            for k in ["logs", "queue", "converted", "converted_paths", "summary_versions", "magic_outputs", "timeline_events", "doc_metrics"]:
                st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else {}
            st.session_state.run_id = None
            st.session_state.active_summary_id = None
            st.session_state.stage = "Idle"
            log("Session outputs cleared.", level="INFO")


def render_logs():
    st.subheader("Live Log")
    logs = st.session_state.logs[-300:]
    txt = "\n".join([f"[{e['time']}][{e['level']}][{e['stage']}][{e.get('doc_id','')}] {e['msg']}" for e in logs])
    st.code(txt or "(no logs yet)", language="text")


def render_prompt_editor():
    st.subheader("Prompt Studio (Editable)")
    pt = st.session_state.prompt_templates
    with st.expander("Summary Prompts", expanded=False):
        pt["summary_system"] = st.text_area("Summary System Prompt", pt["summary_system"], height=140)
        pt["summary_user"] = st.text_area("Summary User Prompt Template", pt["summary_user"], height=220)
        pt["summary_adjust_length"] = st.text_area("Summary Adjust-Length Prompt Template", pt["summary_adjust_length"], height=160)
        pt["summary_chat"] = st.text_area("Summary Chat Prompt Template", pt["summary_chat"], height=180)
    with st.expander("Magic Prompts", expanded=False):
        pt["magic_system"] = st.text_area("Magic System Prompt", pt["magic_system"], height=120)
        pt["magic_user"] = st.text_area("Magic User Prompt Template", pt["magic_user"], height=140)
    with st.expander("Note Keeper Prompts", expanded=False):
        pt["note_organize"] = st.text_area("Note Organize Prompt Template", pt["note_organize"], height=140)
        pt["note_magic"] = st.text_area("Note Magic Prompt Template", pt["note_magic"], height=180)
    with st.expander("agents.yaml Standardization Prompt", expanded=False):
        pt["agents_standardize"] = st.text_area("agents.yaml Standardize Prompt Template", pt["agents_standardize"], height=220)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset Prompts to Defaults"):
            st.session_state.prompt_templates = default_prompt_templates()
            log("Prompt templates reset to defaults.", level="INFO")
            st.rerun()
    with c2:
        st.caption("Tip: keep constraints explicit (word count, Markdown-only, structure).")


# ----------------------------
# Main Pipeline UI: Convert & Summarize
# ----------------------------

def add_files_to_queue(uploaded_files: List) -> None:
    for uf in uploaded_files:
        name = uf.name
        ext = os.path.splitext(name)[1].lower()
        fid = hashlib.sha1((name + str(time.time())).encode("utf-8")).hexdigest()[:10]
        entry = {
            "id": fid,
            "name": name,
            "ext": ext,
            "size": uf.size,
            "status": "Pending",
            "bytes": None,
            "text": None,
        }
        if ext == ".pdf":
            entry["bytes"] = uf.read()
        elif ext in (".txt", ".md"):
            entry["text"] = uf.read().decode("utf-8", errors="replace")
        else:
            log(f"Skipped unsupported upload: {name}", level="WARN", stage="Ingestion")
            continue
        st.session_state.queue.append(entry)
        log(f"Queued file: {name}", stage="Ingestion")


def scan_folder_to_queue(folder_path: str, allowed_exts: Tuple[str, ...] = (".pdf", ".txt", ".md")):
    # Folder paths only work if accessible inside the container.
    if not folder_path or not folder_path.strip():
        st.error("Please provide a folder path.")
        return
    folder_path = folder_path.strip()
    if not os.path.isdir(folder_path):
        st.error("Folder path not found inside this runtime. On Hugging Face Spaces, use multi-upload instead.")
        log(f"Folder path not accessible: {folder_path}", level="ERROR", stage="Ingestion")
        return

    files = []
    for fn in os.listdir(folder_path):
        full = os.path.join(folder_path, fn)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext in allowed_exts:
            files.append(full)

    if not files:
        st.warning("No supported files found in directory.")
        return

    for full in sorted(files):
        name = os.path.basename(full)
        ext = os.path.splitext(name)[1].lower()
        fid = hashlib.sha1((full + str(time.time())).encode("utf-8")).hexdigest()[:10]
        entry = {
            "id": fid,
            "name": name,
            "ext": ext,
            "size": os.path.getsize(full),
            "status": "Pending",
            "bytes": None,
            "text": None,
            "source_path": full,
        }
        if ext == ".pdf":
            with open(full, "rb") as f:
                entry["bytes"] = f.read()
        else:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                entry["text"] = f.read()
        st.session_state.queue.append(entry)
        log(f"Queued from folder: {name}", stage="Ingestion")


def render_queue_table():
    if not st.session_state.queue:
        st.write("Queue is empty.")
        return
    df = pd.DataFrame([{
        "id": q["id"],
        "name": q["name"],
        "type": q["ext"],
        "size_kb": int(q["size"] / 1024),
        "status": q.get("status", ""),
    } for q in st.session_state.queue])
    st.dataframe(df, use_container_width=True, hide_index=True)


def run_conversion_and_summary():
    reset_stop()

    if not st.session_state.queue:
        st.error("No documents in queue.")
        return

    # Start new run
    run_id = new_run_id()
    st.session_state.run_id = run_id
    paths = ensure_dirs(run_id)

    set_stage("Scanning")
    log(f"Run started: {run_id}", stage="Scanning")

    # Write meta snapshot
    meta = {
        "run_id": run_id,
        "created_at": now_iso(),
        "output_language": st.session_state.output_language,
        "ui_language": st.session_state.ui_language,
        "theme": st.session_state.theme,
        "models": st.session_state.model_by_feature,
        "model_global": st.session_state.model_global,
        "queue": [{"name": q["name"], "ext": q["ext"], "size": q["size"], "id": q["id"]} for q in st.session_state.queue],
    }
    st.session_state.run_meta = meta
    write_run_meta(paths, meta)

    # Convert each doc
    set_stage("Converting")
    progress = st.progress(0)
    n = len(st.session_state.queue)

    for i, fe in enumerate(st.session_state.queue):
        if should_stop():
            log("Conversion halted by user.", level="WARN", stage="Converting")
            break
        fid = fe["id"]
        try:
            fe["status"] = "Converting"
            md, metrics, out_path = convert_document(fe, paths)
            st.session_state.converted[fid] = md
            st.session_state.converted_paths[fid] = out_path
            st.session_state.doc_metrics[fid] = metrics
            fe["status"] = "Converted"
            log(f"Converted OK: {fe['name']}", level="SUCCESS", doc_id=fid, stage="Converting")
        except Exception as e:
            fe["status"] = "Failed"
            log(f"Convert failed: {fe['name']} — {e}", level="ERROR", doc_id=fid, stage="Converting")
        progress.progress(int(((i + 1) / n) * 100))

    # Save logs so far
    write_logs(paths)

    # Summarize
    if should_stop():
        set_stage("Interrupted")
        write_logs(paths)
        st.warning("Run interrupted. Partial outputs are available for export.")
        return

    converted_ok = {fid: md for fid, md in st.session_state.converted.items()}
    if not converted_ok:
        st.error("No documents converted successfully; cannot summarize.")
        return

    set_stage("Summarizing")
    log("Building sources block for summarization", stage="Summarizing")
    sources_block = build_sources_block(st.session_state.converted, st.session_state.queue)

    system_p = st.session_state.prompt_templates["summary_system"]
    user_p = st.session_state.prompt_templates["summary_user"].format(
        output_language=st.session_state.output_language,
        sources_block=sources_block,
    ).replace("{run_id}", run_id)  # in case template includes run_id literal

    model = st.session_state.model_by_feature.get("summary", st.session_state.model_global)
    st.subheader("Integrated Summary (streaming)")
    log(f"Calling LLM for summary (model={model})", stage="Summarizing")
    try:
        result = run_llm(system_p, user_p, model=model, stream_to_ui=True)
        summary_md = result.text.strip()
        summary_md = ensure_summary_length(summary_md, st.session_state.output_language, model=model, max_rounds=2)
        sum_path = save_summary(paths, summary_md)
        log(f"Summary saved → {sum_path}", level="SUCCESS", stage="Summarizing")
        add_summary_version("Integrated Summary (v1)", summary_md, {"model": model, "provider": result.provider, "run_id": run_id})
    except Exception as e:
        log(f"Summary generation failed: {e}", level="ERROR", stage="Summarizing")
        st.error(f"Summary generation failed: {e}")
        write_logs(paths)
        return

    write_run_meta(paths, st.session_state.run_meta)
    write_logs(paths)

    set_stage("Completed")
    log("Run completed successfully.", level="SUCCESS", stage="Completed")
    st.success("Completed: per-document Markdown created + integrated summary generated.")


# ----------------------------
# Summary Interaction UI
# ----------------------------

def render_summary_panel():
    st.subheader("Summary Version Vault")
    versions = st.session_state.summary_versions
    if not versions:
        st.write("No summary yet.")
        return

    options = [f"{v['created_at']} — {v['name']} ({v['id']})" for v in versions]
    selected = st.selectbox("Select active summary version", options, index=len(options) - 1)
    sid = selected.split("(")[-1].split(")")[0]
    st.session_state.active_summary_id = sid

    active = get_active_summary_text()
    st.markdown(active)

    # Download active summary
    st.download_button(
        "Download Active Summary.md",
        data=active.encode("utf-8"),
        file_name="summary.md",
        mime="text/markdown",
    )

    # Chat / follow-up prompt
    st.divider()
    st.subheader("Keep Prompting on Summary")
    user_prompt = st.text_area("Prompt", "", height=100, placeholder="Ask for refinement, expansions, or targeted extraction...")
    if st.button("Run Summary Chat"):
        if not user_prompt.strip():
            st.error("Please enter a prompt.")
        else:
            set_stage("Chat")
            log("Running summary chat", stage="Chat")
            model = st.session_state.model_by_feature.get("summary_chat", st.session_state.model_global)
            system_p = "You answer using the summary as context."
            user_p = st.session_state.prompt_templates["summary_chat"].format(
                output_language=st.session_state.output_language,
                summary=active,
                user_prompt=user_prompt,
            )
            st.subheader("Chat Response (streaming)")
            result = run_llm(system_p, user_p, model=model, stream_to_ui=True)
            out = result.text.strip()
            add_summary_version("Chat refinement", out, {"model": model, "provider": result.provider})
            log("Summary chat completed", level="SUCCESS", stage="Chat")

    # WOW Magics
    st.divider()
    st.subheader("WOW Summary Magics (6)")
    c1, c2 = st.columns([2, 1])
    with c1:
        magic = st.selectbox("Choose a magic", WOW_SUMMARY_MAGICS)
    with c2:
        style = st.session_state.painter_style

    if st.button("Apply Magic"):
        run_summary_magic(magic, style)


# ----------------------------
# Agents YAML Manager UI
# ----------------------------

def render_agents_yaml_manager():
    st.subheader("agents.yaml Manager")
    st.write("Upload / standardize / edit / download agents.yaml. Non-standard YAML can be transformed.")

    up = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
    if up is not None:
        txt = up.read().decode("utf-8", errors="replace")
        st.session_state.agents_yaml_text = txt
        log("agents.yaml uploaded.", stage="agents.yaml")

    st.session_state.agents_yaml_text = st.text_area(
        "agents.yaml (editable)",
        st.session_state.agents_yaml_text,
        height=260,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Deterministic Standardize"):
            std, rep = deterministic_standardize_agents_yaml(st.session_state.agents_yaml_text)
            st.session_state.agents_yaml_text = std
            st.session_state.agents_yaml_standard_report = rep
            log("Deterministic standardization completed.", stage="agents.yaml")
    with c2:
        if st.button("LLM Standardize"):
            std, rep = llm_standardize_agents_yaml(st.session_state.agents_yaml_text)
            st.session_state.agents_yaml_text = std
            st.session_state.agents_yaml_standard_report = rep
            log("LLM standardization completed.", stage="agents.yaml")
    with c3:
        st.download_button(
            "Download agents.yaml",
            data=(st.session_state.agents_yaml_text or "").encode("utf-8"),
            file_name="agents.yaml",
            mime="text/yaml",
        )

    if st.session_state.agents_yaml_standard_report:
        st.info(st.session_state.agents_yaml_standard_report)


# ----------------------------
# AI Note Keeper UI
# ----------------------------

def render_note_keeper():
    st.subheader("AI Note Keeper")
    note_input = st.text_area("Paste text/markdown", "", height=200, placeholder="Paste your note here...")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Organize into Markdown"):
            run_note_organize(note_input)
    with c2:
        st.caption("Output language follows the global Output Language selector.")

    st.divider()
    st.subheader("Note Magics (6)")
    magic = st.selectbox("Choose a note magic", NOTE_MAGICS)
    keywords = ""
    color = "yellow"
    if magic == "AI Keywords Highlighter":
        keywords = st.text_input("Keywords (comma-separated)", "")
        color = st.selectbox("Highlight color", ["yellow", "pink", "cyan", "lime", "orange"], index=0)

    if st.button("Apply Note Magic"):
        # Prefer latest note version if exists; else use pasted input
        base_note = st.session_state.note_versions[-1]["text"] if st.session_state.note_versions else note_input
        run_note_magic(base_note, magic, keywords=keywords, color=color)

    st.divider()
    st.subheader("Note Versions")
    if not st.session_state.note_versions:
        st.write("No note outputs yet.")
    else:
        opts = [f"{v['created_at']} — {v['name']} ({v['id']})" for v in st.session_state.note_versions]
        sel = st.selectbox("Select note version", opts, index=len(opts) - 1)
        nid = sel.split("(")[-1].split(")")[0]
        nv = next(v for v in st.session_state.note_versions if v["id"] == nid)
        st.markdown(nv["text"])
        st.download_button("Download Note.md", nv["text"].encode("utf-8"), file_name="note.md", mime="text/markdown")


# ----------------------------
# Dashboard UI
# ----------------------------

def render_dashboard():
    st.subheader("WOW Dashboard & Visualizations")

    # Quick metrics
    q = st.session_state.queue
    converted_count = sum(1 for x in q if x.get("status") == "Converted")
    failed_count = sum(1 for x in q if x.get("status") == "Failed")
    st.write(f"Queue: {len(q)} | Converted: {converted_count} | Failed: {failed_count}")

    # Visualization selector
    viz = st.selectbox("Visualization", VISUALIZATION_OPTIONS)
    if viz == "Run Timeline (Gantt-like)":
        viz_run_timeline()
    elif viz == "Source Contribution Heatmap (Heuristic)":
        viz_source_contribution_heatmap()
    elif viz == "Topic Map (Top Terms)":
        viz_topic_map()
    elif viz == "Entity Network (Heuristic Co-occurrence)":
        viz_entity_network()
    elif viz == "Risk & Uncertainty Radar (Heuristic)":
        viz_risk_radar()
    elif viz == "Version Diff Viewer (Text)":
        viz_version_diff()

    st.divider()
    st.subheader("Per-Document Metrics")
    if st.session_state.doc_metrics:
        df = pd.DataFrame([
            {"doc_id": fid, **m} for fid, m in st.session_state.doc_metrics.items()
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.write("No metrics yet.")

    st.divider()
    st.subheader("Export Packager (ZIP)")
    artifacts_dir = st.session_state.last_run_artifacts_dir
    if artifacts_dir and os.path.isdir(artifacts_dir):
        if st.button("Build Export ZIP"):
            zbytes = build_export_zip(artifacts_dir)
            st.download_button(
                "Download Export Bundle.zip",
                data=zbytes,
                file_name=f"{st.session_state.run_id or 'run'}_bundle.zip",
                mime="application/zip",
            )
    else:
        st.write("No run artifacts directory found yet.")


# ----------------------------
# Main App
# ----------------------------

def main():
    _init_state()
    render_header()
    render_sidebar_settings()

    tabs = st.tabs([
        "Convert & Summarize",
        "Queue & Outputs",
        "WOW Studio",
        "AI Note Keeper",
        "agents.yaml Manager",
        "Dashboard",
        "Logs",
    ])

    # --- Convert & Summarize ---
    with tabs[0]:
        st.subheader("Input Documents")

        mode = st.radio("Ingestion mode", ["Multi-upload (recommended on Spaces)", "Folder path (server-side)"], index=0)
        if mode.startswith("Multi-upload"):
            uploaded = st.file_uploader(
                "Upload multiple documents (txt, md, pdf)",
                type=["txt", "md", "pdf"],
                accept_multiple_files=True,
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Add uploaded files to queue"):
                    if uploaded:
                        add_files_to_queue(uploaded)
                    else:
                        st.error("No files uploaded.")
            with c2:
                if st.button("Clear queue"):
                    st.session_state.queue = []
                    st.session_state.converted = {}
                    st.session_state.converted_paths = {}
                    st.session_state.doc_metrics = {}
                    log("Queue cleared by user.", stage="Ingestion")
        else:
            folder = st.text_input("Folder path (must exist inside this runtime)", value="")
            if st.button("Scan folder and add to queue"):
                scan_folder_to_queue(folder)

        st.divider()
        st.subheader("Queue")
        render_queue_table()

        st.divider()
        st.subheader("Execute")
        st.write("This will: (1) convert each document to Markdown, (2) generate an integrated 3000–4000-word summary.")
        if st.button("Run Conversion + Integrated Summary"):
            run_conversion_and_summary()

        st.divider()
        render_summary_panel()

    # --- Queue & Outputs ---
    with tabs[1]:
        st.subheader("Queue Management & Outputs")
        render_queue_table()

        st.divider()
        st.subheader("Converted Markdown Outputs")
        if st.session_state.converted:
            for fe in st.session_state.queue:
                fid = fe["id"]
                if fid not in st.session_state.converted:
                    continue
                with st.expander(f"{fe['name']} → Markdown", expanded=False):
                    st.caption(f"Saved path: {st.session_state.converted_paths.get(fid, '(memory only)')}")
                    st.markdown(st.session_state.converted[fid])
                    st.download_button(
                        f"Download {slugify(os.path.splitext(fe['name'])[0])}.md",
                        data=st.session_state.converted[fid].encode("utf-8"),
                        file_name=f"{slugify(os.path.splitext(fe['name'])[0])}.md",
                        mime="text/markdown",
                        key=f"dl_{fid}",
                    )
        else:
            st.write("No converted outputs yet.")

        st.divider()
        st.subheader("Summary Versions")
        if st.session_state.summary_versions:
            st.write(f"{len(st.session_state.summary_versions)} summary version(s) available.")
        else:
            st.write("No summary yet.")

    # --- WOW Studio ---
    with tabs[2]:
        render_prompt_editor()

    # --- AI Note Keeper ---
    with tabs[3]:
        render_note_keeper()

    # --- agents.yaml Manager ---
    with tabs[4]:
        render_agents_yaml_manager()

    # --- Dashboard ---
    with tabs[5]:
        render_dashboard()

    # --- Logs ---
    with tabs[6]:
        render_logs()


if __name__ == "__main__":
    main()
