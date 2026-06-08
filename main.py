import os
from pypdf import PdfReader
import sentence_transformers
import math

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text: str, chunk_size: int = 256, chunk_overlap: int = 32):
    """
    Chunking engine 
    following are the tasks: text, chunks array in which all the "chunk" will be stored. 
    start_idx, end_idx.
    """
    words = text.split()
    chunks = []
    start_idx = 0
    while start_idx < len(words):
        end_idx = start_idx + chunk_size
        chunk_words = words[start_idx:end_idx]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        start_idx += chunk_size - chunk_overlap
    return chunks

def load_pdf(folder_path:str):
    document_content = []
    if not os.path.exists(folder_path):
        print("Folder Not found")
        return None

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            print(f"processing {filename}")

            try:
                reader = PdfReader(file_path)
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()

                    page_chunks = chunk_text(page_text)

                    for small_chunk in page_chunks:
                        chunk = {
                            "text":small_chunk,
                            "source":filename,
                            "page":page_num
                        }
                        document_content.append(chunk)
                
            except Exception as e:
                print(f"following error occured: {e}")
    return document_content


def vector_embeddings(text):
    embeddings = model.encode(text)
    return embeddings

def embedding_to_db(chunks_list):
    vector_db = []
    print("\n Generating In-memory Vector_DB")
    for chunk in chunks_list:
        raw_text = chunk["text"]
        source = chunk["source"]
        page = chunk["page"]
    
        vector_embedding = vector_embeddings(raw_text)

        updated_chunk = {
            "text":raw_text,
            "source":source,
            "page":page,
            "vector":vector_embedding # new data added.
        }
        vector_db.append(updated_chunk)
    return vector_db

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    # 1. calculate dot product
    dot_product = math.sumprod(v1,v2)
    # 2. calculate magnitude for v1
    v1_mag = math.hypot(*v1)
    # 3. calculate magnitude for v2
    v2_mag = math.hypot(*v2)
    # 4. return cosine similarity.
    if v1_mag == 0.0 or v2_mag == 0:
        return 0.0
    else:
        cos_sim = dot_product / (v1_mag * v2_mag)
        return cos_sim

def retrieve(user_query:str, vector_db:list, top_k:int = 3):
    query_vector = vector_embeddings(user_query)
    scored_chunks = []

    for chunk in vector_db:
        chunk_vector = chunk["vector"]

        score = cosine_similarity(chunk_vector, query_vector)

        scored_chunks.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": score
        })

        sorted_chunks = sorted(scored_chunks, key=lambda x:x["score"], reverse=True)

    return sorted_chunks[:top_k]



KNOWLEDGE_FOLDER = "./knowledge"
final_chunks = load_pdf(KNOWLEDGE_FOLDER)
if final_chunks:
    print("Ingestion successfull")
    print(f"Total pieces created: {len(final_chunks)}")
    print("building vector database")
    vector_db = embedding_to_db(final_chunks)
    print(f"Length of the vector db: {len(vector_db)}")
    # print(f"first vector in the db: {vector_db[0]}")
    print(f"First chunk in vector db: {vector_db[0]["text"]}")
    if vector_db:
        query = "What is a critical threat or prompt injection?"
        results = retrieve(query, vector_db, top_k=2)
        print(f"\n--- RETRIEVAL RESULTS FOR: '{query}' ---")
        for i, res in enumerate(results, start=1):
            print(f"\nMatch #{i} (Score: {res['score']:.4f})")
            print(f"Source: {res['source']} | Page: {res['page']}")
            print(f"Text Preview: {res['text'][:150]}...")
else:
    print("No chunks were created. check if your folder path is correct or not.")
    