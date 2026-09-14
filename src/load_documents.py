from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Location of our Tolkien lore document
file_path = "data/raw/aragorn.txt"

# Load the document
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

print("Number of documents loaded:", len(documents))

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50
)

# Split the document into smaller chunks
chunks = text_splitter.split_documents(documents)

print("Number of chunks created:", len(chunks))

# Display every chunk
for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {i} ---\n")
    print(chunk.page_content)
    print("\nMetadata:", chunk.metadata)