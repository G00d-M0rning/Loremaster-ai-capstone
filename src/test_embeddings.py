from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load a small local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Our test question
question = "Who was Aragorn's wife?"

# Two possible pieces of lore
lore_1 = "Aragorn married Arwen, daughter of Elrond."
lore_2 = "Sauron was the Dark Lord of Mordor."

# Convert the text into numerical vectors
question_vector = embeddings.embed_query(question)
lore_vectors = embeddings.embed_documents([lore_1, lore_2])

# Compare the question with each piece of lore
score_1 = cosine_similarity(
    [question_vector],
    [lore_vectors[0]]
)[0][0]

score_2 = cosine_similarity(
    [question_vector],
    [lore_vectors[1]]
)[0][0]

print("\nQuestion:")
print(question)

print("\nLore 1:")
print(lore_1)
print("Similarity score:", round(score_1, 3))

print("\nLore 2:")
print(lore_2)
print("Similarity score:", round(score_2, 3))