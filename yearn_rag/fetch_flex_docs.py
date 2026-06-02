from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Iterable
from urllib import request
from urllib.parse import urlparse


LLMS_URL = "https://flexmeow.com/llms.txt"
USER_AGENT = "ysupport-rag/1.0"
OUTPUT_DIR = Path(__file__).resolve().parent / "flex-docs"


class _FlexMainContentParser(HTMLParser):
    _SKIP_CLASSES = {
        "header-bar",
        "footer-bar",
        "logo-container",
        "wallet-box",
        "tooltip-text",
        "tooltip-icon",
        "inner-border",
    }
    _BLOCK_TAGS = {"p", "div", "main", "ul", "ol", "table", "tr"}
    _HEADING_TAGS = {"h1", "h2", "h3", "h4"}
    _TEXT_TAGS = {"p", "li", "span", "strong", "em", "code", "a", "td", "th"}
    _VOID_TAGS = {"img", "br", "meta", "link", "input", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture_main_depth = 0
        self._skip_depth = 0
        self._current_href: str | None = None
        self._list_depth = 0
        self._buffer: list[str] = []
        self._lines: list[str] = []
        self._active_heading: str | None = None
        self._active_text_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {key: value or "" for key, value in attrs}
        classes = set(attrs_map.get("class", "").split())
        if tag == "main":
            self._capture_main_depth += 1
        if self._capture_main_depth <= 0:
            return
        if self._skip_depth > 0:
            if tag not in self._VOID_TAGS:
                self._skip_depth += 1
            return
        if classes & self._SKIP_CLASSES or tag in {"script", "style", "svg"}:
            self._skip_depth = 1
            return
        if tag == "img":
            self._buffer.append(" ")
            return
        if tag in {"ul", "ol"}:
            self._flush_buffer()
            self._list_depth += 1
        elif tag == "li":
            self._flush_buffer()
            indent = "  " * max(self._list_depth - 1, 0)
            self._buffer.append(f"{indent}- ")
            self._active_text_tag = "li"
        elif tag in self._HEADING_TAGS:
            self._flush_buffer()
            self._active_heading = tag
        elif tag == "a":
            self._current_href = attrs_map.get("href") or None
        elif tag in {"br"}:
            self._buffer.append("\n")
        elif tag in self._TEXT_TAGS:
            self._active_text_tag = tag

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._capture_main_depth <= 0 or self._skip_depth > 0:
            return
        if tag == "br":
            self._buffer.append("\n")
        elif tag == "img":
            self._buffer.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "main" and self._capture_main_depth > 0:
            self._flush_buffer()
            self._capture_main_depth -= 1
            return
        if self._capture_main_depth <= 0:
            return
        if self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "a":
            self._current_href = None
        elif tag in self._HEADING_TAGS:
            self._flush_buffer(as_heading=tag)
            self._active_heading = None
        elif tag == "li":
            self._flush_buffer()
            self._active_text_tag = None
        elif tag in {"ul", "ol"}:
            self._flush_buffer()
            self._list_depth = max(0, self._list_depth - 1)
        elif tag in self._BLOCK_TAGS:
            self._flush_buffer()
        elif tag in self._TEXT_TAGS:
            self._active_text_tag = None

    def handle_data(self, data: str) -> None:
        if self._capture_main_depth <= 0 or self._skip_depth > 0:
            return
        normalized = re.sub(r"\s+", " ", data)
        if not normalized.strip():
            return
        if self._current_href and self._active_text_tag == "a":
            self._buffer.append(f"{normalized.strip()} ({self._current_href})")
            return
        self._buffer.append(normalized)

    def get_markdown(self) -> str:
        self._flush_buffer()
        compacted: list[str] = []
        previous_blank = False
        for line in self._lines:
            stripped = line.rstrip()
            if not stripped:
                if not previous_blank:
                    compacted.append("")
                previous_blank = True
                continue
            compacted.append(stripped)
            previous_blank = False
        return "\n".join(compacted).strip()

    def _flush_buffer(self, *, as_heading: str | None = None) -> None:
        if not self._buffer:
            return
        text = unescape("".join(self._buffer))
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +", " ", text).strip()
        self._buffer.clear()
        if not text:
            return
        if as_heading:
            heading_level = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}[as_heading]
            self._lines.extend([f"{heading_level} {text}", ""])
            return
        self._lines.extend([text, ""])


def fetch_text(url: str) -> str:
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    with request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_llms_links(llms_text: str) -> list[str]:
    return re.findall(r"\((https://[^)]+)\)", llms_text)


def extract_title(html_text: str, fallback: str) -> str:
    match = re.search(r"<title>\s*(.*?)\s*</title>", html_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return fallback
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title or fallback


def render_markdown_document(*, title: str, source_url: str, body: str) -> str:
    safe_title = title.replace('"', '\\"')
    slug = urlparse(source_url).path or "/"
    return (
        "---\n"
        f'title: "{safe_title}"\n'
        f'slug: "{slug}"\n'
        f'source_url: "{source_url}"\n'
        "---\n\n"
        f"# {title}\n\n"
        f"{body.strip()}\n"
    )


def write_markdown_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def slug_from_url(url: str) -> str:
    if url == LLMS_URL:
        return "llms"
    suffix = url.rstrip("/").rsplit("/", 1)[-1]
    return suffix or "index"


def build_fetch_targets() -> list[dict[str, str]]:
    llms_text = fetch_text(LLMS_URL)
    urls = parse_llms_links(llms_text)
    high_value_urls = [url for url in urls if url in {
        "https://flexmeow.com/docs",
        "https://flexmeow.com/risks",
        "https://flexmeow.com/info",
    }]
    targets = [
        {"url": LLMS_URL, "title": "Flex LLMs Index", "filename": "llms.md"},
    ]
    for url in high_value_urls:
        targets.append(
            {
                "url": url,
                "title": "",
                "filename": f"{slug_from_url(url)}.md",
            }
        )
    return targets


def convert_html_page(url: str) -> str:
    html_text = fetch_text(url)
    fallback_title = f"Flex {slug_from_url(url).replace('-', ' ').title()}".strip()
    title = extract_title(html_text, fallback_title)
    parser = _FlexMainContentParser()
    parser.feed(html_text)
    body = parser.get_markdown()
    if not body:
        body = _extract_simple_panel_markdown(html_text)
    return render_markdown_document(title=title, source_url=url, body=body)


def convert_llms_index(url: str) -> str:
    llms_text = fetch_text(url).strip()
    body = llms_text if llms_text else "# Flex\n"
    return render_markdown_document(
        title="Flex LLMs Index",
        source_url=url,
        body=body,
    )


def fetch_all_targets(targets: Iterable[dict[str, str]]) -> None:
    for target in targets:
        url = target["url"]
        output_path = OUTPUT_DIR / target["filename"]
        if url == LLMS_URL:
            content = convert_llms_index(url)
        else:
            content = convert_html_page(url)
        write_markdown_file(output_path, content)
        print(f"✅ Saved {url} -> {output_path.name}")


def _extract_simple_panel_markdown(html_text: str) -> str:
    match = re.search(
        r'<div class="simple-panel">(.*?)(?=<div class="footer-bar"|</main>)',
        html_text,
        re.IGNORECASE | re.DOTALL,
    )
    fragment = match.group(1) if match else html_text
    fragment = re.sub(r"<script\b[^>]*>.*?</script>", "", fragment, flags=re.IGNORECASE | re.DOTALL)
    fragment = re.sub(r"<style\b[^>]*>.*?</style>", "", fragment, flags=re.IGNORECASE | re.DOTALL)

    def _replace_anchor(match_obj: re.Match[str]) -> str:
        href = match_obj.group(1).strip()
        label = _strip_tags(match_obj.group(2))
        label = re.sub(r"\s+", " ", unescape(label)).strip()
        if not label:
            return ""
        if href.startswith("#"):
            return label
        return f"{label} ({href})"

    replacements = [
        (r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", _replace_anchor),
        (r"<h2\b[^>]*>(.*?)</h2>", lambda m: f"\n## {_clean_inline_html(m.group(1))}\n"),
        (r"<h3\b[^>]*>(.*?)</h3>", lambda m: f"\n### {_clean_inline_html(m.group(1))}\n"),
        (r"<h4\b[^>]*>(.*?)</h4>", lambda m: f"\n#### {_clean_inline_html(m.group(1))}\n"),
        (r"<li\b[^>]*>(.*?)</li>", lambda m: f"\n- {_clean_inline_html(m.group(1))}"),
        (r"<p\b[^>]*>(.*?)</p>", lambda m: f"\n{_clean_inline_html(m.group(1))}\n"),
        (r"<br\s*/?>", lambda _: "\n"),
    ]
    for pattern, replacement in replacements:
        fragment = re.sub(pattern, replacement, fragment, flags=re.IGNORECASE | re.DOTALL)

    fragment = re.sub(r"</?(div|span|ul|ol|strong|em|code|table|tbody|thead|tr|td|th)\b[^>]*>", "", fragment, flags=re.IGNORECASE)
    fragment = _strip_tags(fragment)
    fragment = unescape(fragment)
    fragment = re.sub(r"[ \t]+\n", "\n", fragment)
    fragment = re.sub(r"\n{3,}", "\n\n", fragment)
    return fragment.strip()


def _clean_inline_html(text: str) -> str:
    text = _strip_tags(text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = build_fetch_targets()
    fetch_all_targets(targets)


if __name__ == "__main__":
    main()
