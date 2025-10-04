import os
import re
import json
import tiktoken

LOCAL_DOCS_DIR = "yearn-devdocs/docs"
OUTPUT_JSON = "cleaned_yearn_docs.json"

META_CONTEXT_FILE = "llm_meta_context.md"

CHUNK_MAX_TOKENS = 750
CHUNK_OVERLAP_TOKENS = 150
TOKEN_ENCODING_MODEL = "cl100k_base"

EXCLUDED_FOLDERS = [
    "developers/smart-contracts/deprecated",
    "resources",
    "partners", 
    "unused",
    "contributing/operations/decision-making.md",
    "contributing/operations/osc-levels.md"
]

def sanitize_for_id(text: str) -> str:
    """
    Sanitizes a string to be a valid ASCII ID by removing non-ASCII characters,
    replacing spaces and slashes with hyphens, and converting to lowercase.
    """
    # Replace slashes and spaces with hyphens
    text = text.replace('/', '-').replace('\\', '-').replace(' ', '-')
    # Encode to ASCII, ignoring characters that can't be converted, then decode back to a string
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining problematic characters (like quotes)
    text = re.sub(r'[^\w\-\.]', '', text)
    # Convert to lowercase
    return text.lower()

def clean_markdown(text):
    """
    Cleans markdown text for RAG processing by removing noise while PRESERVING
    valuable markdown structure like code fences, lists, and links.
    """
    # 1a. Remove YAML frontmatter
    text = re.sub(r'^---[\s\S]*?---\n', '', text, flags=re.MULTILINE)
    # 1b. Remove Docusaurus/Next.js import statements
    text = re.sub(r'^import.*from.*;?\n', '', text, flags=re.MULTILINE)

    # 1. Remove excessively long bytecode, which is pure noise for an LLM
    text = re.sub(r"0x[a-fA-F0-9]{500,}", "[BYTECODE REMOVED]", text)

    # 2. Remove common large ABI JSON blocks, which are better handled by other tools
    text = re.sub(r'\[\s*\{\s*"inputs":.*?\}\s*\]', "[ABI ARRAY REMOVED]", text, flags=re.DOTALL)
    text = re.sub(r'\{\s*"abi":\s*\[.*?\],\s*"bytecode":.*?\}', "[CONTRACT ARTIFACT REMOVED]", text, flags=re.DOTALL)

    # 3. Add helpful markers for specific, known table types without altering the table itself
    text = re.sub(r"(\|\s*Contract Name\s*\|\s*Contract Address\s*\|)", r"[CONTRACT TABLE]\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(\|\s*Description\s*\|\s*Address\s*\|)", r"[CONTRACT TABLE]\n\1", text, flags=re.IGNORECASE)

    # 4. Strip basic HTML tags that might have slipped into the markdown
    text = re.sub(r'<[^>]+>', '', text)

    # 5. Normalize whitespace: collapse multiple blank lines into one for token efficiency
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def chunk_text_by_tokens(text, model=TOKEN_ENCODING_MODEL, max_tokens=500, overlap_tokens=100):
    """
    Splits text into overlapping chunks based on token count in a robust and safe manner.
    
    :param text: The input text to chunk
    :param model: The token encoding model (e.g., cl100k_base for OpenAI embeddings)
    :param max_tokens: Maximum tokens per chunk
    :param overlap_tokens: Number of tokens to overlap between chunks
    :return: List of text chunks
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
        # 1. SAFE END CALCULATION: Ensure the end index does not exceed the list length.
        end = min(start + max_tokens, total_tokens)
        
        chunk_tokens = tokens[start:end]
        
        try:
            chunk_text = encoding.decode(chunk_tokens)
            # Append the chunk if it's not just whitespace
            if chunk_text.strip():
                chunks.append(chunk_text)
        except Exception as e:
            print(f"⚠️ Error decoding tokens: {e}. Skipping this chunk.")

        # Calculate the start of the next chunk
        next_start = start + max_tokens - overlap_tokens
        
        # 2. INFINITE LOOP GUARD: Ensure we always make progress.
        if next_start <= start:
            start += 1 # Force progress by at least one token
        else:
            start = next_start
            
    return chunks

def process_markdown_files():
    """Reads and processes all markdown files with intelligent heading detection."""
    documents = []
    
    # --- Process the special meta context file (logic is the same) ---
    if os.path.exists(META_CONTEXT_FILE):
        print(f"Processing special meta context file: {META_CONTEXT_FILE}")
        with open(META_CONTEXT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        clean_text_data = content.strip() 
        chunks = chunk_text_by_tokens(clean_text_data, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
        for i, chunk in enumerate(chunks):
            documents.append({
                "filename": "internal_context.md",
                "doc_title": "Internal AI Context",
                "section_heading": "System Knowledge",
                "source_path": META_CONTEXT_FILE,
                "chunk_id": i,
                "text": chunk
            })
        print(f"✅ Added {len(chunks)} chunks from the meta context file.")
    else:
        print(f"⚠️ Meta context file '{META_CONTEXT_FILE}' not found. Skipping.")

    # --- Process the main docs directory with stateful heading tracking ---
    print(f"Processing main documentation directory: {LOCAL_DOCS_DIR}")
    for root, _, files in os.walk(LOCAL_DOCS_DIR):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                
                try: 
                    relative_filepath = os.path.relpath(filepath, LOCAL_DOCS_DIR).replace('\\', '/')

                    # Exclusion
                    excluded = False
                    for exclusion in EXCLUDED_FOLDERS:
                        if relative_filepath == exclusion or relative_filepath.startswith(exclusion + '/'):
                            excluded = True
                            break
                    if excluded:
                        continue

                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                
                    # Title Extraction
                    lines = content.split('\n')
                    doc_title = os.path.splitext(file)[0].replace('-', ' ').title()
                    for line in lines:
                        if line.startswith('# '):
                            doc_title = line[2:].strip()
                            break
                
                    # Heading & Content Tracking
                    current_h2 = ""
                    current_h3 = ""
                    current_h4 = ""

                    sections = re.split(r'(^##\s+.*$|^###\s+.*$|^####\s+.*$)', content, flags=re.MULTILINE)

                    if sections[0].strip():
                        process_section(documents, sections[0].strip(), file, relative_filepath, doc_title, "", "", "", "", "")

                    for i in range(1, len(sections), 2):
                        heading = sections[i].strip()
                        section_content = sections[i+1].strip()
                    
                        parent_h2 = ""
                        parent_h3 = ""

                        if heading.startswith('## '):
                            current_h2 = heading[3:].strip()
                            current_h3 = ""
                            current_h4 = ""
                        elif heading.startswith('### '):
                            current_h3 = heading[4:].strip()
                            current_h4 = ""
                            parent_h2 = current_h2 # An H3's parent is the current H2
                        elif heading.startswith('#### '):
                            current_h4 = heading[5:].strip()
                            parent_h2 = current_h2 # An H4's parent is the current H2...
                            parent_h3 = current_h3 # ...and the current H3
                        
                        process_section(documents, section_content, file, relative_filepath, doc_title, current_h2, current_h3, current_h4, parent_h2, parent_h3)
                except Exception as e:
                    print(f"❌ Error processing file {filepath}: {e}. Skipping.")
                    continue
    return documents

def process_section(documents, section_content, filename, source_path, doc_title, h2, h3, h4, parent_h2="", parent_h3=""):
    """
    Helper function with final, structure-based filtering and context-injection.
    """
    if not section_content:
        return
    
    if h4:
        section_heading = f"{h2} > {h3} > {h4}" if h3 else f"{h2} > {h4}"
    elif h3:
        section_heading = f"{h2} > {h3}"
    elif h2:
        section_heading = h2
    else:
        section_heading = doc_title

    clean_text = clean_markdown(section_content)
    
    # 1. Smarter Filtering
    is_code_or_table = '```' in clean_text or '|' in clean_text
    if len(clean_text.split()) < 7 and not is_code_or_table:
        return
    lines = clean_text.strip().split('\n')
    link_lines = [line for line in lines if re.match(r'^\s*[\*\-]\s*\[.*\]\(.*\)\s*$', line)]
    if len(lines) >= 2 and (len(link_lines) / len(lines)) > 0.7:
        return

    # 2. Structure-Based Context Injection
    final_text = clean_text
    # A chunk is a subsection if it has an H3 or H4 heading.
    if h3 or h4:
        # The immediate parent is H3 if it exists, otherwise it's H2.
        parent_context_heading = parent_h3 if parent_h3 else parent_h2
        if parent_context_heading:
            final_text = f"This content is part of the '{parent_context_heading}' section.\n\n---\n\n{clean_text}"
        
    chunks = chunk_text_by_tokens(final_text, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    
    for i, chunk in enumerate(chunks):
        documents.append({
            "filename": filename,
            "doc_title": doc_title,
            "section_heading": section_heading,
            "source_path": source_path,
            "chunk_id": f"{sanitize_for_id(section_heading)}-{i}",
            "text": chunk
        })

def main():
    """
    Main execution function to process documentation files and save the output.
    """
    print("--- Starting Documentation Processing ---")
    
    yearn_docs = process_markdown_files()
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(yearn_docs, f, indent=2, ensure_ascii=False)
    
    print(f"\n--- Summary ---")
    print(f"✅ Successfully processed and saved {len(yearn_docs)} chunks to {OUTPUT_JSON}.")

if __name__ == "__main__":
    main()
