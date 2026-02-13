import json
import time
import re
import os
import hashlib
from pathlib import Path

import openai
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException 

# API Keys (loaded from .env one level up)
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "docs")
PINECONE_HOST = os.getenv("PINECONE_HOST")
CLEAR_NAMESPACE = os.getenv("CLEAR_NAMESPACE", "0") == "1"
EMBEDDING_RETRIES = 3

EMBEDDING_SOURCES = {
    "docs_and_internal": {
        "input_json": "cleaned_yearn_docs.json",
        "namespace": "yearn-docs" # The "Current Truth" namespace
    },
    "yips": {
        "input_json": "cleaned_yips.json",
        "namespace": "yearn-yips" # The "Historical Context" namespace
    }
}

# Initialize OpenAI & Pinecone
try:
    openai.api_key = OPENAI_API_KEY
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
    index.describe_index_stats() 
    print("✅ Successfully connected to Pinecone index.")
except Exception as e:
    print(f"❌ Failed to initialize clients or connect to Pinecone index: {e}")
    exit(1)

def clear_namespace(namespace):
    """
    Safely deletes all vectors from a specific namespace.
    RAISES an exception if deletion cannot be confirmed.
    """
    try:
        print("--- Clearing Namespace ---")
        stats = index.describe_index_stats()
        # Add a print statement to see exactly what the API returns
        print(f"Initial stats response: {stats}") 
        
        namespace_stats = stats.get("namespaces", {}).get(namespace)
        
        if namespace_stats and namespace_stats.get("vector_count", 0) > 0:
            vector_count = namespace_stats["vector_count"]
            print(f"Found {vector_count} vectors in namespace '{namespace}'. Deleting old data...")
            index.delete(delete_all=True, namespace=namespace) 
            
            print("Waiting for deletion to propagate (15 seconds)...") # Increased wait time
            time.sleep(15) 
            
            print("Verifying deletion...")
            # Loop a few times to give the API time to update
            for attempt in range(3):
                stats_after = index.describe_index_stats()
                count_after = stats_after.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
                if count_after == 0:
                    print(f"✅ Namespace '{namespace}' cleared successfully.")
                    return # Success, exit the function
                print(f"Verification attempt {attempt + 1}: Found {count_after} vectors remaining. Waiting 5 more seconds...")
                time.sleep(5)
            
            # If the loop finishes and count is still not 0, it's a failure.
            raise Exception(f"Failed to clear namespace '{namespace}'. {count_after} vectors remain after deletion attempt.")
        else:
            print(f"ℹ️ No vectors found in namespace '{namespace}'. Skipping deletion.")
            
    except Exception as e:
        print(f"❌ An unrecoverable error occurred while clearing namespace '{namespace}': {e}")
        # Re-raise the exception to HALT the script
        raise e

def sanitize_for_id(text: str) -> str:
    text = text.replace('/', '-').replace('\\', '-').replace(' ', '-')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\-\.]', '', text)
    return text.lower()

def generate_embedding(texts):
    """Generates OpenAI embeddings for a list of texts (batched)."""
    if not texts:
        return []
    print(f"🔹 Generating embeddings for a batch of {len(texts)} texts...")
    for attempt in range(EMBEDDING_RETRIES):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=texts,
                encoding_format="float"
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            wait_s = 2 ** attempt
            print(f"❌ Error during embedding generation (attempt {attempt + 1}/{EMBEDDING_RETRIES}): {e}")
            if attempt < EMBEDDING_RETRIES - 1:
                time.sleep(wait_s)
    return None

def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def process_and_embed_source(config):
    """
    Loads, embeds, and upserts documents for a single configured source.
    """
    input_json = config["input_json"]
    namespace = config["namespace"]
    
    print(f"\n--- Processing Source: {input_json} -> Namespace: {namespace} ---")

    # 1. Load docs for this source
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            docs_to_process = json.load(f)
        if not docs_to_process:
            print("⚠️ Input file is empty. Nothing to process for this source.")
            return 0 # Return 0 vectors processed
        print(f"✅ Loaded {len(docs_to_process)} document chunks from {input_json}.")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_json}'. Skipping this source.")
        return 0
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{input_json}'. Skipping this source.")
        return 0

    # 2. Clear the target namespace if requested
    if CLEAR_NAMESPACE:
        clear_namespace(namespace)
    else:
        print("ℹ️ CLEAR_NAMESPACE not set; skipping namespace wipe.")

    # 3. Batch process and upsert
    total_vectors_upserted = 0
    batch_size = 100
    for i in range(0, len(docs_to_process), batch_size):
        batch_docs = docs_to_process[i:i + batch_size]
        texts = [doc["text"] for doc in batch_docs]
        
        embeddings = generate_embedding(texts)
        if embeddings is None:
            print(f"   Skipping batch {i//batch_size + 1} due to embedding failure.")
            continue

        vectors = []
        for doc, embedding in zip(batch_docs, embeddings):
            # --- UPDATED METADATA CREATION ---
            metadata = {
                "text": doc.get("text", ""),
                "filename": doc.get("filename", "unknown"),
                "doc_title": doc.get("doc_title", "Unknown Title"),
                "section_heading": doc.get("section_heading", "Unknown Section"),
                "source_path": doc.get("source_path", "unknown"),
                "chunk_id": doc.get("chunk_id", -1),
                "chunk_index": doc.get("chunk_index", doc.get("chunk_id", -1)),
                "doc_id": doc.get("doc_id"),
                "doc_last_modified": doc.get("doc_last_modified"),
                "source_type": doc.get("source_type"),
                "source_url": doc.get("source_url"),
                "content_hash": content_hash(doc.get("text", "")),
                # Add YIP fields using .get() - they will be None if not present
                "yip_status": doc.get("yip_status"),
                "yip_number": doc.get("yip_number"),
                "yip_created": doc.get("yip_created"),
                "yip_discussion_link": doc.get("yip_discussion_link")
            }
            # --- NEW: Filter out keys with None values ---
            # Pinecone doesn't accept None, so we remove these keys entirely.
            final_metadata = {k: v for k, v in metadata.items() if v is not None}
            
            id_base = final_metadata.get("doc_id") or final_metadata.get("source_path", "unknown")
            sanitized_path = sanitize_for_id(str(id_base))
            sanitized_chunk_id = sanitize_for_id(str(final_metadata.get("chunk_index", final_metadata.get("chunk_id", -1))))
            
            vectors.append({
                "id": f"{sanitized_path}-{sanitized_chunk_id}", 
                "values": embedding,
                "metadata": final_metadata # Use the filtered metadata
            })
        
        try:
            upsert_response = index.upsert(vectors=vectors, namespace=namespace)
            upserted_count = upsert_response.get('upserted_count', 0)
            total_vectors_upserted += upserted_count
            print(f"✅ Upserted batch {i//batch_size + 1}: Stored {upserted_count} vectors into namespace '{namespace}'.")
        except PineconeApiException as e:
            print(f"❌ Pinecone API Error during upsert in batch {i//batch_size + 1}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error during upsert in batch {i//batch_size + 1}: {e}")
        
        time.sleep(0.2)
    
    return total_vectors_upserted

# --- Main Execution Block ---
if __name__ == "__main__":
    grand_total_upserted = 0
    
    # Loop through the configured sources and process each one
    for source_name, config in EMBEDDING_SOURCES.items():
        grand_total_upserted += process_and_embed_source(config)

    print("\n--- Summary ---")
    print(f"✅ Process complete. Successfully stored a grand total of {grand_total_upserted} document chunks across all sources.")
