import os
from pypdf import PdfReader
import sentence_transformers

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
            "vector_embedding":vector_embedding # new data added.
        }
        vector_db.append(updated_chunk)
    return vector_db

def cosine_similarity(v1: list, v2: list):
    # 1. calculate dot product
    # 2. calculate magnitude for v1
    # 3. calculate magnitude for v2
    # 4. return cosine similarity.
    pass


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


else:
    print("No chunks were created. check if your folder path is correct or not.")
    