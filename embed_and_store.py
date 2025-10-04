import json
import time
import re
import openai
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException 

# API Keys
OPENAI_API_KEY = "-"
PINECONE_API_KEY = "-"
PINECONE_INDEX_NAME = "-"
PINECONE_HOST = "-"
NAMESPACE = "-"
INPUT_JSON = "cleaned_yearn_docs.json"

# Initialize OpenAI & Pinecone
try:
    openai.api_key = OPENAI_API_KEY
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
    index.describe_index_stats() 
    print("✅ Successfully connected to Pinecone index.")
except Exception as e:
    print(f"❌ Failed to initialize clients or connect to Pinecone index: {e}")
    exit(1) # Exit if we can't connect

def clear_namespace(namespace):
    """Safely deletes all vectors from a specific namespace by waiting and verifying."""
    try:
        stats = index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get(namespace)
        
        if namespace_stats and namespace_stats.get("vector_count", 0) > 0:
            vector_count = namespace_stats["vector_count"]
            print(f"Found {vector_count} vectors in namespace '{namespace}'. Deleting old data...")
            index.delete(delete_all=True, namespace=namespace) 
            print("Waiting for deletion to propagate (10 seconds)...")
            time.sleep(10) 
            
            print("Verifying deletion...")
            stats_after = index.describe_index_stats()
            count_after = stats_after.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
            
            if count_after == 0:
                 print(f"✅ Namespace '{namespace}' cleared successfully.")
            else:
                 print(f"⚠️ WARNING: Namespace '{namespace}' clear command was issued, but {count_after} vectors are still reported. Deletion may be slow or have failed.")
        else:
            print(f"ℹ️ No vectors found in namespace '{namespace}'. Skipping deletion.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while clearing namespace '{namespace}': {e}")

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
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=texts,
            encoding_format="float"
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"❌ Error during embedding generation for this batch: {e}")
        return None
    
# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load cleaned docs with error handling
    try:
        print(f"Loading documents from {INPUT_JSON}...")
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            yearn_docs = json.load(f)
        if not yearn_docs:
            print("⚠️ Input file is empty. Nothing to process.")
            exit(0)
        print(f"✅ Loaded {len(yearn_docs)} document chunks.")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{INPUT_JSON}'. Please run process_docs.py first.")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{INPUT_JSON}'. The file may be corrupted.")
        exit(1)

    # 2. Clear the namespace
    clear_namespace(NAMESPACE)

    # 3. Batch process and upsert
    total_vectors_upserted = 0
    batch_size = 100
    print("\n--- Starting Embedding and Upsert Process ---")
    for i in range(0, len(yearn_docs), batch_size):
        batch_docs = yearn_docs[i:i + batch_size]
        texts = [doc["text"] for doc in batch_docs]
        
        embeddings = generate_embedding(texts)
        if embeddings is None:
            print(f"   Skipping batch {i//batch_size + 1} due to embedding failure.")
            continue

        vectors = []
        for doc, embedding in zip(batch_docs, embeddings):
            metadata = {
                "text": doc.get("text", ""),
                "filename": doc.get("filename", "unknown"),
                "doc_title": doc.get("doc_title", "Unknown Title"),
                "section_heading": doc.get("section_heading", "Unknown Section"),
                "source_path": doc.get("source_path", "unknown"),
                "chunk_id": doc.get("chunk_id", -1)
            }
            sanitized_path = sanitize_for_id(metadata['source_path'])
            sanitized_chunk_id = sanitize_for_id(str(metadata['chunk_id']))
            vectors.append({
                "id": f"{sanitized_path}-{sanitized_chunk_id}", 
                "values": embedding,
                "metadata": metadata
            })
        
        try:
            upsert_response = index.upsert(vectors=vectors, namespace=NAMESPACE)
            # Use the response to get the actual count
            upserted_count = upsert_response.get('upserted_count', 0)
            total_vectors_upserted += upserted_count
            print(f"✅ Upserted batch {i//batch_size + 1}: Stored {upserted_count} vectors.")
        except PineconeApiException as e:
            print(f"❌ Pinecone API Error during upsert in batch {i//batch_size + 1}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error during upsert in batch {i//batch_size + 1}: {e}")
        
        time.sleep(0.2)

    print("\n--- Summary ---")
    print(f"✅ Process complete. Successfully stored {total_vectors_upserted} out of {len(yearn_docs)} document chunks in Pinecone.")
