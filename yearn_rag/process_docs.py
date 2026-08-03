import os
import re
import json
import tiktoken
import frontmatter
import hashlib
from datetime import datetime, timezone
from pathlib import Path

DOC_SOURCES = {
    "yearn": {
        "source_dir": "yearn-devdocs/docs",
        "output_json": "cleaned_yearn_docs.json",
        "excluded_folders": [
            "developers/smart-contracts/deprecated",
            "resources",
            "partners",
            "unused",
            "contributing/operations/decision-making.md",
            "contributing/operations/osc-levels.md"
        ],
        "base_url": "https://docs.yearn.fi",
        "route_base_path": "/",
        "source_type": "documentation"
    },
    "flex": {
        "source_dir": "flex-docs",
        "output_json": "cleaned_flex_docs.json",
        "excluded_folders": [],
        "base_url": "https://flexmeow.com",
        "route_base_path": "/",
        "source_type": "documentation"
    }
}

YIP_SOURCE_PATH_RE = re.compile(
    r"^contributing/governance/yips/yip-(\d+)\.(?:md|mdx)$",
    re.IGNORECASE,
)


class YipMetadataError(ValueError):
    pass


CHUNK_MAX_TOKENS = 750
CHUNK_OVERLAP_TOKENS = 150
TOKEN_ENCODING_MODEL = "cl100k_base"

def _markdown_table_value(content, field_name):
    match = re.search(
        rf"^\|\s*{re.escape(field_name)}\s*\|\s*(.*?)\s*\|$",
        content,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def _strip_markdown_emphasis(value):
    return value.strip().strip("*_ ") if value else value


def _markdown_link_url(value):
    if not value:
        return None
    match = re.search(r"\[[^\]]+\]\(([^)]+)\)", value)
    return match.group(1) if match else None


def extract_yip_metadata(source_path, content):
    """
    Returns structured metadata for a canonical Yearn YIP document.
    """
    path_match = YIP_SOURCE_PATH_RE.fullmatch(source_path or "")
    if not path_match:
        return None

    yip_number = int(path_match.group(1))
    table_number = _markdown_table_value(content or "", "YIP")
    status = _strip_markdown_emphasis(
        _markdown_table_value(content or "", "Outcome")
    )
    created_date = _strip_markdown_emphasis(
        _markdown_table_value(content or "", "Created")
    )
    discussion_link = _markdown_link_url(
        _markdown_table_value(content or "", "Forum discussion")
    )
    if table_number is None or not table_number.isdigit():
        raise YipMetadataError(f"Missing YIP number metadata in {source_path}")
    if int(table_number) != yip_number:
        raise YipMetadataError(
            f"YIP metadata mismatch in {source_path}: table says {table_number}"
        )
    missing_fields = [
        field
        for field, value in (
            ("Outcome", status),
            ("Created", created_date),
            ("Forum discussion", discussion_link),
        )
        if not value
    ]
    if missing_fields:
        raise YipMetadataError(
            f"Missing YIP metadata in {source_path}: {', '.join(missing_fields)}"
        )

    return {
        "yip_number": yip_number,
        "status": status,
        "created_date": created_date,
        "discussion_link": discussion_link,
    }


def sanitize_for_id(text: str) -> str:
    """
    Sanitizes a string to be a valid ASCII ID by removing non-ASCII characters,
    replacing spaces and slashes with hyphens, and converting to lowercase.
    """
    text = text.replace('/', '-').replace('\\', '-').replace(' ', '-')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\-\.]', '', text)
    return text.lower()

def stable_doc_id(source_path: str) -> str:
    """
    Returns a stable, ASCII doc id derived from the source path.
    """
    if not source_path:
        return ""
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()
    return digest

def iso_utc_from_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

def normalize_route_base_path(route_base_path):
    if not route_base_path:
        return "/"
    rbp = route_base_path.strip()
    if not rbp.startswith('/'):
        rbp = '/' + rbp
    if rbp != '/' and rbp.endswith('/'):
        rbp = rbp.rstrip('/')
    return rbp or "/"

def derive_doc_path(relative_path):
    doc_path = relative_path.replace('\\', '/')
    doc_path = re.sub(r'\.(mdx|md)$', '', doc_path, flags=re.IGNORECASE)
    if doc_path.endswith('/index'):
        doc_path = doc_path[:-len('/index')]
    if doc_path == 'index':
        doc_path = ''
    if not doc_path:
        return '/'
    return '/' + doc_path.lstrip('/')

def join_url(base_url, path):
    if not base_url or not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return base_url.rstrip('/') + '/' + path.lstrip('/')

def build_source_url(relative_path, metadata, base_url, route_base_path):
    if not base_url:
        return None
    rbp = normalize_route_base_path(route_base_path)

    path = None
    if metadata:
        path = metadata.get("permalink") or metadata.get("slug")

    if path:
        if not path.startswith('/'):
            path = rbp.rstrip('/') + '/' + path
    else:
        doc_path = derive_doc_path(relative_path)
        if rbp != '/':
            if doc_path == '/':
                path = rbp
            else:
                path = rbp.rstrip('/') + doc_path
        else:
            path = doc_path

    path = re.sub(r'//+', '/', path)
    if path != '/' and path.endswith('/'):
        path = path.rstrip('/')
    return join_url(base_url, path)

def strip_yaml_frontmatter(text):
    """
    Removes YAML frontmatter only if it appears at the very top of the file.
    This avoids stripping content between horizontal rules later in the doc.
    """
    match = re.match(r'\s*---\s*\n([\s\S]*?)\n---\s*(?:\n|$)', text)
    if match and ':' in match.group(1):
        return text[match.end():]
    return text

def strip_top_mdx_imports(text):
    """
    Removes top-of-file MDX import/export blocks without touching code fences.
    """
    lines = text.splitlines()
    if not lines:
        return text

    out = []
    in_top = True
    skipping_block = False

    for line in lines:
        stripped = line.strip()
        if in_top:
            if skipping_block:
                if stripped.endswith(';'):
                    skipping_block = False
                    continue
                if stripped and 'from ' not in stripped and not stripped.endswith(',') and not stripped.startswith(('{', '}', ')')):
                    skipping_block = False
                    in_top = False
                    out.append(line)
                continue
            if stripped == "":
                out.append(line)
                continue
            if stripped.startswith("import ") or stripped.startswith("export "):
                if not stripped.endswith(';'):
                    skipping_block = True
                continue
            in_top = False
            out.append(line)
        else:
            out.append(line)

    return "\n".join(out)

def strip_html_outside_code(text):
    """
    Strips basic HTML tags outside fenced code blocks.
    """
    parts = re.split(r"(```[\s\S]*?```)", text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'<[^>]+>', '', parts[i])
    return ''.join(parts)

def convert_mdx_links_outside_code(text):
    """
    Converts MDX/HTML links to "label (url)" outside fenced code blocks
    so URLs survive HTML stripping.
    """
    parts = re.split(r"(```[\s\S]*?```)", text)
    link_patterns = [
        (r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', r'\2 (\1)'),
        (r'<Link\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</Link>', r'\2 (\1)')
    ]
    for i in range(0, len(parts), 2):
        for pattern, repl in link_patterns:
            parts[i] = re.sub(pattern, repl, parts[i], flags=re.IGNORECASE | re.DOTALL)
    return ''.join(parts)

def clean_markdown(text):
    """
    Cleans markdown text for RAG processing by removing noise while PRESERVING
    valuable markdown structure like code fences, lists, and links.
    """
    text = strip_yaml_frontmatter(text)
    text = strip_top_mdx_imports(text)

    text = re.sub(r"0x[a-fA-F0-9]{500,}", "[BYTECODE REMOVED]", text)
    text = re.sub(r'\[\s*\{\s*"inputs":.*?\}\s*\]', "[ABI ARRAY REMOVED]", text, flags=re.DOTALL)
    text = re.sub(r'\{\s*"abi":\s*\[.*?\],\s*"bytecode":.*?\}', "[CONTRACT ARTIFACT REMOVED]", text, flags=re.DOTALL)

    text = re.sub(r"(\|\s*Contract Name\s*\|\s*Contract Address\s*\|)", r"[CONTRACT TABLE]\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(\|\s*Description\s*\|\s*Address\s*\|)", r"[CONTRACT TABLE]\n\1", text, flags=re.IGNORECASE)

    text = convert_mdx_links_outside_code(text)
    text = strip_html_outside_code(text)

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def chunk_text_by_tokens(text, model=TOKEN_ENCODING_MODEL, max_tokens=500, overlap_tokens=100):
    """
    Splits text into overlapping chunks based on token count in a robust and safe manner.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be less than max_tokens")

    try:
        encoding = tiktoken.get_encoding(model)
        tokens = encoding.encode(text)
    except Exception as e:
        print(f"⚠️ Error encoding text with tiktoken: {e}. Skipping chunking for this text.")
        return []

    chunks = []
    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]

        try:
            chunk_text = encoding.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
        except Exception as e:
            print(f"⚠️ Error decoding tokens: {e}. Skipping this chunk.")

        next_start = start + max_tokens - overlap_tokens
        if next_start <= start:
            start += 1
        else:
            start = next_start

    return chunks

def chunk_text_by_lines(text, model=TOKEN_ENCODING_MODEL, max_tokens=500, overlap_tokens=100):
    """
    Builds chunks by whole lines to avoid splitting URLs/addresses/tables mid-line.
    Falls back to token chunking for extremely long lines.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be less than max_tokens")

    try:
        encoding = tiktoken.get_encoding(model)
    except Exception as e:
        print(f"⚠️ Error loading tiktoken encoding: {e}. Falling back to token chunking.")
        return chunk_text_by_tokens(text, model=model, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    lines = text.splitlines()
    chunks = []
    current_lines = []
    current_tokens = 0

    def line_token_count(line):
        try:
            return len(encoding.encode(line + "\n"))
        except Exception:
            return len(encoding.encode(line))

    def build_overlap(lines_to_overlap):
        if overlap_tokens <= 0:
            return [], 0
        overlap_lines = []
        overlap_count = 0
        for line in reversed(lines_to_overlap):
            overlap_lines.append(line)
            overlap_count += line_token_count(line)
            if overlap_count >= overlap_tokens:
                break
        overlap_lines.reverse()
        return overlap_lines, overlap_count

    for line in lines:
        line_tokens = line_token_count(line)
        if line_tokens > max_tokens:
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            subchunks = chunk_text_by_tokens(line, model=model, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            chunks.extend([c for c in subchunks if c.strip()])
            current_lines = []
            current_tokens = 0
            continue

        if current_tokens + line_tokens > max_tokens and current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_lines, current_tokens = build_overlap(current_lines)

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks

def process_markdown_files(source_dir, excluded_folders, source_type="documentation", base_url=None, route_base_path="/"):
    """
    Reads and processes all markdown files from a given source directory.
    This function is now generic and driven by the configuration.
    """
    documents = []
    print(f"\n--- Processing documentation source: {source_dir} ---")

    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise RuntimeError(f"Documentation source directory is missing: {source_dir}")

    processed_files = 0

    for root, dirs, files in os.walk(source_dir):
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith((".md", ".mdx")):
                filepath = os.path.join(root, file)
                try:
                    relative_filepath = os.path.relpath(filepath, source_dir).replace('\\', '/')

                    is_excluded = False
                    for exclusion in excluded_folders:
                        if relative_filepath == exclusion or relative_filepath.startswith(exclusion + '/'):
                            is_excluded = True
                            break
                    if is_excluded:
                        continue

                    post = frontmatter.load(filepath)
                    processed_files += 1
                    content = post.content
                    raw_metadata = post.metadata

                    default_title_from_filename = os.path.splitext(file)[0].replace('-', ' ').title()
                    yip_metadata = extract_yip_metadata(relative_filepath, content)
                    document_source_type = "yip" if yip_metadata else source_type
                    source_url = build_source_url(relative_filepath, raw_metadata, base_url, route_base_path)
                    if not source_url:
                        source_url = raw_metadata.get("discussion_link") or raw_metadata.get("discussions-to")

                    doc_title = raw_metadata.get("title") or raw_metadata.get("sidebar_label")
                    if not doc_title:
                        h1_match = re.search(r"^\s*#\s+(.+)", content, re.MULTILINE)
                        if h1_match:
                            doc_title = h1_match.group(1).strip()
                        else:
                            doc_title = default_title_from_filename

                    doc_last_modified = iso_utc_from_mtime(os.path.getmtime(filepath))

                    file_chunk_counter = 0
                    current_h2, current_h3, current_h4 = "", "", ""

                    sections = re.split(r'(^##\s+.*$|^###\s+.*$|^####\s+.*$)', content, flags=re.MULTILINE)

                    if sections[0].strip():
                        file_chunk_counter = process_section(
                            documents,
                            sections[0].strip(),
                            file,
                            relative_filepath,
                            doc_title,
                            "",
                            "",
                            "",
                            "",
                            "",
                            file_chunk_counter,
                            yip_metadata=yip_metadata,
                            source_type=document_source_type,
                            source_url=source_url,
                            doc_last_modified=doc_last_modified
                        )

                    for i in range(1, len(sections), 2):
                        heading = sections[i].strip()
                        section_content = sections[i+1].strip()

                        parent_h2, parent_h3 = "", ""

                        if heading.startswith('## '):
                            current_h2 = heading[3:].strip()
                            current_h3, current_h4 = "", ""
                        elif heading.startswith('### '):
                            current_h3 = heading[4:].strip()
                            current_h4 = ""
                            parent_h2 = current_h2
                        elif heading.startswith('#### '):
                            current_h4 = heading[5:].strip()
                            parent_h2, parent_h3 = current_h2, current_h3

                        file_chunk_counter = process_section(
                            documents,
                            section_content,
                            file,
                            relative_filepath,
                            doc_title,
                            current_h2,
                            current_h3,
                            current_h4,
                            parent_h2,
                            parent_h3,
                            file_chunk_counter,
                            yip_metadata=yip_metadata,
                            source_type=document_source_type,
                            source_url=source_url,
                            doc_last_modified=doc_last_modified
                        )

                except YipMetadataError:
                    raise
                except Exception as e:
                    raise RuntimeError(
                        f"Error processing documentation file {filepath}: {e}"
                    ) from e
    if processed_files == 0 or not documents:
        raise RuntimeError(
            f"Documentation source produced no usable chunks: {source_dir}"
        )
    return documents


def write_text_if_changed(path: Path, content: str) -> bool:
    try:
        if path.read_text(encoding="utf-8") == content:
            return False
    except FileNotFoundError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
    return True

def process_section(documents, section_content, filename, source_path, doc_title, h2, h3, h4, parent_h2, parent_h3, file_chunk_counter, yip_metadata, source_type, source_url, doc_last_modified):
    """
    Helper function with final, structure-based filtering and context-injection.
    """
    if not section_content:
        return file_chunk_counter

    if h4:
        section_heading = f"{h2} > {h3} > {h4}" if h3 else f"{h2} > {h4}"
    elif h3:
        section_heading = f"{h2} > {h3}"
    elif h2:
        section_heading = h2
    else:
        section_heading = doc_title

    clean_text = clean_markdown(section_content)

    is_code_or_table = '```' in clean_text or '|' in clean_text
    if len(clean_text.split()) < 7 and not is_code_or_table:
        return file_chunk_counter

    lines = clean_text.strip().split('\n')
    link_lines = [line for line in lines if re.match(r'^\s*[\*\-]\s*\[.*\]\(.*\)\s*$', line)]
    if len(lines) >= 2 and (len(link_lines) / len(lines)) > 0.7:
        return file_chunk_counter

    final_text = clean_text
    if h3 or h4:
        parent_context_heading = parent_h3 if parent_h3 else parent_h2
        if parent_context_heading:
            final_text = f"This content is part of the '{parent_context_heading}' section.\n\n---\n\n{clean_text}"

    context_heading = doc_title
    if section_heading and section_heading != doc_title:
        context_heading = f"{doc_title} > {section_heading}"
    final_text = f"{context_heading}\n\n{final_text}"

    chunks = chunk_text_by_lines(final_text, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)

    doc_id = stable_doc_id(source_path)

    for chunk in chunks:
        chunk_id = file_chunk_counter

        doc_object = {
            "filename": filename,
            "doc_title": doc_title,
            "section_heading": section_heading,
            "source_path": source_path,
            "chunk_id": chunk_id,
            "chunk_index": chunk_id,
            "doc_id": doc_id,
            "doc_last_modified": doc_last_modified,
            "text": chunk,
            "source_type": source_type,
            "source_url": source_url
        }

        if yip_metadata:
            doc_object["yip_number"] = yip_metadata.get("yip_number")
            if yip_metadata.get("status") is not None:
                doc_object["yip_status"] = yip_metadata.get("status")
            if yip_metadata.get("created_date") is not None:
                doc_object["yip_created"] = yip_metadata.get("created_date")
            if yip_metadata.get("discussion_link") is not None:
                doc_object["yip_discussion_link"] = yip_metadata.get("discussion_link")

        documents.append(doc_object)
        file_chunk_counter += 1

    return file_chunk_counter

def main():
    """
    Main execution function to process all configured documentation sources.
    """
    print("--- Starting All Documentation Processing ---")

    for source_name, config in DOC_SOURCES.items():
        source_type = config.get("source_type", "documentation")
        base_url = config.get("base_url")
        route_base_path = config.get("route_base_path", "/")
        final_docs = process_markdown_files(
            config["source_dir"],
            config["excluded_folders"],
            source_type=source_type,
            base_url=base_url,
            route_base_path=route_base_path
        )

        output_text = json.dumps(final_docs, indent=2, ensure_ascii=False) + "\n"
        output_path = Path(config["output_json"])
        output_changed = write_text_if_changed(output_path, output_text)
        action = "Saved" if output_changed else "Unchanged"
        print(
            f"✅ {action} {len(final_docs)} total chunks for "
            f"'{source_name}' in {config['output_json']}."
        )

if __name__ == "__main__":
    main()
