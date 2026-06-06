def chunk_text(text: str, chunk_size: int = 256, chunk_overlap: int = 32):
    """
    Chunks text by words with a sliding window overlap.
    """
    words = text.split()
    chunks = []
    
    start_index = 0
    while start_index < len(words):
        # Determine the end index of the current chunk
        end_index = start_index + chunk_size
        
        # Extract chunk and join with spaces
        chunk_words = words[start_index:end_index]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        # Move the sliding window
        start_index += chunk_size - chunk_overlap
    print(chunks)
    return chunks

# Example Usage:
document_text = "Your long document goes here " * 50
pieces = chunk_text(document_text, chunk_size=256, chunk_overlap=32)

print(f"Total pieces created: {len(pieces)}")
print(f"First piece word count: {len(pieces[0].split())}")