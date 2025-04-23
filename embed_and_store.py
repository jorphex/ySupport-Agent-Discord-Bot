import json
import time
import traceback
import openai
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException, NotFoundException 

# Config
OPENAI_API_KEY = "key"
PINECONE_API_KEY = "key"
PINECONE_HOST = "host"  
PINECONE_INDEX_NAME = "index" 
EMBEDDING_SOURCES = {
    "yearn": {
        "input_json": "cleaned_yearn_docs.json",
        "namespace": "yearn" 
    },
    "bearn": {
        "input_json": "cleaned_bearn_docs.json",
        "namespace": "bearn" 
    }
}

EMBEDDING_MODEL = "text-embedding-3-large" 
BATCH_SIZE = 100 

try:
    print("Initializing OpenAI...")
    openai.api_key = OPENAI_API_KEY
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY) 

    print(f"Checking if Pinecone index '{PINECONE_INDEX_NAME}' exists...")

    try:
        pc.describe_index(PINECONE_INDEX_NAME)
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' found.")
      
    except NotFoundException:
        print(f"❌ Error: Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Please create it.")
        exit(1) 
      
    except PineconeApiException as e:
        print(f"❌ Pinecone API error while checking index '{PINECONE_INDEX_NAME}': {e}")
        exit(1)

    print(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
    index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
    print(f"✅ Connected to Pinecone index '{PINECONE_INDEX_NAME}' at host '{PINECONE_HOST}'.")
    print("Fetching index stats...")
    stats = index.describe_index_stats()
    print(stats) 

except Exception as e:
    print(f"❌ Failed during initialization phase: {e}") 
    traceback.print_exc()
    exit(1) 

# Delete everything to prep for fresh data
def clear_namespace(pinecone_index, namespace):
    if not namespace:
        print("⚠️ Namespace cannot be empty. Skipping deletion.")
        return
    try:
        print(f"Attempting to clear namespace '{namespace}'...")
        stats = pinecone_index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get(namespace)

        if namespace_stats and namespace_stats.get("vector_count", 0) > 0:
            vector_count = namespace_stats["vector_count"]
            print(f"Found {vector_count} vectors in namespace '{namespace}'. Deleting old data...")

            pinecone_index.delete(delete_all=True, namespace=namespace) 

            print("Waiting for deletion to propagate (10 seconds)...")
            time.sleep(10) 

            print("Verifying deletion...")
            stats_after = pinecone_index.describe_index_stats()
            count_after = stats_after.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
            if count_after == 0:
                 print(f"✅ Namespace '{namespace}' cleared successfully (vector count is 0).")
            else:
                 print(f"⚠️ Namespace '{namespace}' clear command issued, but {count_after} vectors still reported. Deletion might be ongoing or failed silently.")
        else:
            print(f"ℹ️ No vectors found in namespace '{namespace}'. Skipping deletion.")

    except PineconeApiException as e:
        if hasattr(e, 'status') and e.status == 404: 
             print(f"ℹ️ Namespace '{namespace}' likely doesn't exist or is already empty (API status {e.status}).")
        else:
             print(f"❌ Pinecone API error while clearing namespace '{namespace}': Status={getattr(e, 'status', 'N/A')}, Reason={getattr(e, 'reason', 'N/A')}, Body={getattr(e, 'body', 'N/A')}")
    except Exception as e:
        print(f"❌ Unexpected error while clearing namespace '{namespace}': {e}")
        import traceback
        traceback.print_exc() 

def generate_embedding(texts, model=EMBEDDING_MODEL):
    if not texts:
        return []
    try:
        texts = [str(text) if text is not None else "" for text in texts]
        response = openai.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float"
        )

        print(f"🔹 Generated embeddings for batch of {len(texts)} texts.")
        return [item.embedding for item in response.data]
    except openai.APIError as e:
        print(f"❌ OpenAI API Error during embedding generation: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error during embedding generation: {e}")
        return None

def embed_and_store_source(config, pinecone_index):
    input_json = config["input_json"]
    namespace = config["namespace"]
    print(f"\n--- Embedding source from: {input_json} into namespace: {namespace} ---")

    try:
        with open(input_json, "r", encoding="utf-8") as f:
            docs = json.load(f)
        if not docs:
            print(f"ℹ️ Input file '{input_json}' is empty. Skipping embedding for this source.")
            return 0 
        print(f"Loaded {len(docs)} document chunks from {input_json}.")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {input_json}. Skipping this source.")
        return 0
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from {input_json}: {e}. Skipping this source.")
        return 0
    except Exception as e:
        print(f"❌ Error loading file {input_json}: {e}. Skipping this source.")
        return 0

    clear_namespace(pinecone_index, namespace)

    total_vectors_upserted = 0

    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i:i + BATCH_SIZE]
        texts = [doc.get("text", "") for doc in batch_docs] 

        valid_texts = [text for text in texts if text and isinstance(text, str) and text.strip()]
        valid_indices = [idx for idx, text in enumerate(texts) if text and isinstance(text, str) and text.strip()]

        if not valid_texts:
            print(f"   Skipping batch {i//BATCH_SIZE + 1} as it contains no valid text.")
            continue

        embeddings = generate_embedding(valid_texts)
        if embeddings is None or len(embeddings) != len(valid_texts):
            print(f"   ⚠️ Embedding generation failed or returned incorrect count for batch {i//BATCH_SIZE + 1}. Skipping batch.")
            continue 

        vectors = []
        embedding_idx = 0
        for original_idx, doc in enumerate(batch_docs):
             if original_idx in valid_indices:

                 doc_id = f"{doc.get('source_path', 'unknown')}-{doc.get('chunk_id', 'N/A')}"
                 metadata = {
                     "source_path": doc.get("source_path", "unknown"),
                     "filename": doc.get("filename", "unknown"),
                     "doc_title": doc.get("doc_title", "N/A"),          
                     "section_heading": doc.get("section_heading", "N/A"), 
                     "chunk_id": doc.get("chunk_id", -1),
                     "text": str(doc.get("text", "")) 
                 }
                 vectors.append({
                     "id": doc_id,
                     "values": embeddings[embedding_idx],
                     "metadata": metadata
                 })
                 embedding_idx += 1

        if not vectors:
             print(f"   No vectors constructed for batch {i//BATCH_SIZE + 1}. Skipping upsert.")
             continue

        try:
            upsert_response = pinecone_index.upsert(vectors=vectors, namespace=namespace)
            upserted_count = upsert_response.get('upserted_count', 0)
            total_vectors_upserted += upserted_count
            print(f"✅ Upserted batch {i//BATCH_SIZE + 1}: Stored {upserted_count} vectors into namespace '{namespace}'.")
        except PineconeApiException as e:
            print(f"❌ Pinecone API error during upsert in batch {i//BATCH_SIZE + 1} into namespace '{namespace}': {e}")
        except Exception as e:
            print(f"❌ Unexpected error during upsert in batch {i//BATCH_SIZE + 1} into namespace '{namespace}': {e}")
        time.sleep(0.1)

    print(f"✅ Finished processing for {input_json}. Total vectors upserted to namespace '{namespace}': {total_vectors_upserted}")
    return total_vectors_upserted

if __name__ == "__main__":
    try:
        print("Initializing OpenAI...")
        openai.api_key = OPENAI_API_KEY
        print("Initializing Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)

        print(f"Checking if Pinecone index '{PINECONE_INDEX_NAME}' exists...")
        try:
            pc.describe_index(PINECONE_INDEX_NAME)
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' found.")
        except NotFoundException:
            print(f"❌ Error: Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Please create it.")
            exit(1)
        except PineconeApiException as e:
            print(f"❌ Pinecone API error while checking index '{PINECONE_INDEX_NAME}': {e}")
            exit(1)

        print(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
        index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST) 
        print(f"✅ Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
        stats = index.describe_index_stats()
        print("Index Stats:", stats)

    except Exception as e:
        print(f"❌ Failed during initialization phase: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    grand_total_upserted = 0
    start_time = time.time()

    for source_name, config in EMBEDDING_SOURCES.items():
        grand_total_upserted += embed_and_store_source(config, index)

    end_time = time.time()
    print(f"\n--- Embedding Summary ---")
    print(f"✅ Successfully stored a grand total of {grand_total_upserted} document chunks across all sources.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
