import os
import re
import json
import tiktoken

# Config
DOC_SOURCES = {
    "yearn": {
        "repo_dir": "yearn-devdocs",
        "docs_subdir": "docs",
        "output_json": "cleaned_yearn_docs.json",
        "excluded_folders": [
            "developers/smart-contracts/deprecated",
            "resources",
            "partners",
            "unused",
            "contributing/operations/decision-making.md",
            "contributing/operations/osc-levels.md"
        ]
    },
    "bearn": {
        "repo_dir": "bearn-docs",
        "docs_subdir": "docs",
        "output_json": "cleaned_bearn_docs.json",
        "excluded_folders": [
            # "example/subfolder",
            # "example/file.md"
        ]
    }
}

DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 100
TOKEN_ENCODING_MODEL = "cl100k_base"


def extract_markdown_title(content):
    match = re.search(r"^\s*#\s+(.+)", content, re.MULTILINE)
    title = match.group(1).strip() if match else None
    if title:
        title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', title)
        title = re.sub(r'[*_`]', '', title)
    return title

def clean_markdown(text):
    """ Cleans markdown text for RAG processing. """
    text = re.sub(r"0x[a-fA-F0-9]{500,}", "[BYTECODE REMOVED]", text)
    text = re.sub(r'\[\s*\{\s*"inputs":.*?\}\s*\]', "[ABI ARRAY REMOVED]", text, flags=re.DOTALL)
    text = re.sub(r'\{\s*"abi":\s*\[.*?\],\s*"bytecode":.*?\}', "[CONTRACT ARTIFACT REMOVED]", text, flags=re.DOTALL)
    text = re.sub(r"(\|\s*Contract Name\s*\|\s*Contract Address\s*\|)", r"[CONTRACT TABLE]\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(\|\s*Description\s*\|\s*Address\s*\|)", r"[CONTRACT TABLE]\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_by_headings(markdown_content):
    sections = []
    current_content = []
    current_heading = None

    heading_pattern = re.compile(r"^(#{2,6})\s+(.*)", re.MULTILINE)

    last_match_end = 0
    for match in heading_pattern.finditer(markdown_content):
        section_text = markdown_content[last_match_end:match.start()].strip()
        if section_text or current_heading is None:
             sections.append((current_heading, section_text))

        heading_text = match.group(2).strip()
        heading_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading_text)
        heading_text = re.sub(r'[*_`]', '', heading_text)

        current_heading = heading_text
        last_match_end = match.end()

    final_content = markdown_content[last_match_end:].strip()
    if final_content or current_heading is not None:
        sections.append((current_heading, final_content))

    if not sections and markdown_content:
         sections.append((None, markdown_content))
    elif sections and sections[0][0] is not None and markdown_content.find(heading_pattern.pattern) != 0:
         first_match = heading_pattern.search(markdown_content)
         if first_match and first_match.start() > 0:
             initial_content = markdown_content[:first_match.start()].strip()
             if initial_content:

                 if not sections or sections[0][1] != initial_content:
                     sections.insert(0, (None, initial_content))
         elif not first_match and markdown_content:
             sections = [(None, markdown_content)]

    filtered_sections = [(h, c) for h, c in sections if c]
    if not filtered_sections and sections:
        return sections
    elif not filtered_sections and not sections:
        return []
    else:
        return filtered_sections


def chunk_text_by_tokens(text, model=TOKEN_ENCODING_MODEL, max_tokens=DEFAULT_MAX_TOKENS, overlap_tokens=DEFAULT_OVERLAP_TOKENS):
    """
    Splits text into overlapping chunks based on token count using tiktoken.
    (Implementation remains the same as the previous version)
    """
    if not text:
        return []

    try:
        encoding = tiktoken.get_encoding(model)
    except Exception as e:
        print(f"⚠️ Error getting tiktoken encoding '{model}': {e}. Falling back to basic splitting.")
        avg_chars_per_token = 4
        max_chars = max_tokens * avg_chars_per_token
        overlap_chars = overlap_tokens * avg_chars_per_token
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            next_start = start + max_chars - overlap_chars
            if next_start <= start:
                next_start = start + 1
            start = next_start
        return chunks

    try:
        tokens = encoding.encode(text)
    except Exception as e:
        print(f"⚠️ Error encoding text with tiktoken: {e}. Skipping chunking for this text.")
        return []

    chunks = []
    start_token = 0
    while start_token < len(tokens):
        end_token = min(start_token + max_tokens, len(tokens))
        chunk_tokens = tokens[start_token:end_token]

        try:
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text.strip())
        except Exception as e:
            print(f"⚠️ Error decoding tokens: {e}. Skipping this chunk.")

        next_start_token = start_token + max_tokens - overlap_tokens
        if next_start_token <= start_token:
             next_start_token = start_token + 1
        start_token = min(next_start_token, len(tokens))
        if start_token >= len(tokens) and end_token == len(tokens):
             break

    return [chunk for chunk in chunks if chunk]


def process_markdown_files(docs_base_dir, excluded_folders, source_config):
    """
    Reads, cleans, splits by heading, chunks, and adds metadata (including heading)
    to markdown files.
    """
    all_documents = []
    if not os.path.isdir(docs_base_dir):
        print(f"❌ Error: Documentation directory not found: {docs_base_dir}")
        return all_documents

    print(f"\nProcessing markdown files in: {docs_base_dir}")
    print(f"Excluding: {excluded_folders}")

    max_tokens = source_config.get('max_tokens', DEFAULT_MAX_TOKENS)
    overlap_tokens = source_config.get('overlap_tokens', DEFAULT_OVERLAP_TOKENS)
    print(f"Chunking parameters: max_tokens={max_tokens}, overlap_tokens={overlap_tokens}")

    for root, _, files in os.walk(docs_base_dir):
        files.sort()
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                relative_filepath = os.path.relpath(filepath, docs_base_dir)

                excluded = False
                for exclusion in excluded_folders:
                    if relative_filepath == exclusion or relative_filepath.startswith(exclusion + os.sep):
                        excluded = True
                        break
                if excluded:
                    continue

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        raw_content = f.read()

                    if len(raw_content.strip()) < 5:
                        print(f"   Skipping potentially empty file: {relative_filepath}")
                        continue

                    doc_title = extract_markdown_title(raw_content)
                    title_for_section = doc_title if doc_title else os.path.splitext(file)[0]

                    cleaned_content = clean_markdown(raw_content)
                    if len(cleaned_content) < 5:
                         print(f"   Skipping file (empty after cleaning): {relative_filepath}")
                         continue

                    sections = split_by_headings(cleaned_content)
                    if not sections:
                        print(f"   No content sections found after splitting by heading: {relative_filepath}")

                    file_chunk_counter = 0
                    for section_heading, section_content in sections:
                        if not section_content.strip():
                            continue

                        current_section_heading_meta = section_heading if section_heading else f"Overview ({title_for_section})"

                        chunks = chunk_text_by_tokens(
                            section_content,
                            max_tokens=max_tokens,
                            overlap_tokens=overlap_tokens
                        )

                        if not chunks:

                            continue

                        for chunk_text in chunks:
                            metadata = {
                                "source_path": relative_filepath,
                                "filename": file,
                                "doc_title": doc_title if doc_title else "N/A",
                                "section_heading": current_section_heading_meta,
                                "chunk_id": file_chunk_counter,
                            }
                            all_documents.append({
                                **metadata,
                                "text": chunk_text
                            })
                            file_chunk_counter += 1

                except Exception as e:
                    print(f"❌ Error processing file {filepath}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"Finished processing {docs_base_dir}. Found {len(all_documents)} chunks.")
    return all_documents

if __name__ == "__main__":
    total_chunks_processed = 0
    for source_name, config in DOC_SOURCES.items():
        print(f"\n--- Processing source: {source_name.upper()} ---")

        repo_dir = config["repo_dir"]
        docs_subdir = config["docs_subdir"]
        docs_dir = os.path.join(repo_dir, docs_subdir)

        processed_docs = process_markdown_files(docs_dir, config["excluded_folders"], config)

        if processed_docs:
            output_json_path = config["output_json"]
            try:
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(processed_docs, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved {len(processed_docs)} chunks for {source_name} to {output_json_path}")
                total_chunks_processed += len(processed_docs)
            except IOError as e:
                print(f"❌ Error writing output file {output_json_path}: {e}")
            except Exception as e:
                 print(f"❌ Unexpected error saving JSON for {source_name}: {e}")
        else:
            print(f"ℹ️ No documents processed or chunked for source: {source_name}")

    print(f"\n--- Summary ---")
    print(f"✅ Total chunks processed across all sources: {total_chunks_processed}")
