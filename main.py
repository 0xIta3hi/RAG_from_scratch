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

# Example Usage:
document_text = "Your long document goes here " * 50
pieces = chunk_text(document_text, chunk_size=256, chunk_overlap=32)

print(f"Total pieces created: {len(pieces)}")
print(f"First piece word count: {len(pieces[0].split())}")