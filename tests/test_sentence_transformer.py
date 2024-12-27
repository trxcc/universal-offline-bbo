from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = ["Hello, how are you?", "I am fine, thank you.", "No, I am not fine."]
embeddings = model.encode(text)
print(embeddings)
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
