import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_FOLDER = "data/raw"
DB_FOLDER = "chroma_db"


def read_lore_file(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    # ----------------------------------------------
    # Read article metadata
    # ----------------------------------------------

    title_match = re.search(
        r"^TITLE:\s*(.+)$",
        text,
        re.MULTILINE
    )

    source_match = re.search(
        r"^SOURCE:\s*(.+)$",
        text,
        re.MULTILINE
    )

    title = (
        title_match.group(1).strip()
        if title_match
        else os.path.basename(file_path)
    )

    source_url = (
        source_match.group(1).strip()
        if source_match
        else ""
    )

    # ----------------------------------------------
    # Find every ## SECTION block
    # ----------------------------------------------

    section_pattern = (
        r"## SECTION:\s*(.+?)\n"
        r"(.*?)(?=\n## SECTION:|\Z)"
    )

    matches = re.findall(
        section_pattern,
        text,
        re.DOTALL
    )

    documents = []

    for section_title, section_text in matches:

        section_title = section_title.strip()
        section_text = section_text.strip()

        if not section_text:
            continue

        document = Document(
            page_content=section_text,
            metadata={
                "title": title,
                "section": section_title,
                "source": file_path,
                "url": source_url,
            }
        )

        documents.append(document)

    return documents


# --------------------------------------------------
# 1. Read every article as structured sections
# --------------------------------------------------

documents = []
article_count = 0

for filename in os.listdir(DATA_FOLDER):

    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(
        DATA_FOLDER,
        filename
    )

    article_documents = read_lore_file(
        file_path
    )

    if article_documents:
        article_count += 1
        documents.extend(article_documents)

        print(
            f"Loaded: {filename} "
            f"({len(article_documents)} sections)"
        )

    else:
        print(
            f"WARNING: No sections found in {filename}"
        )


print("\n================================")
print("STRUCTURED LORE LOADED")
print("================================")
print(f"Articles: {article_count}")
print(f"Sections: {len(documents)}")


# --------------------------------------------------
# 2. Split sections into searchable chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=[
        "\n\n",
        ". ",
        " ",
        ""
    ]
)

chunks = text_splitter.split_documents(
    documents
)

print(f"Chunks:   {len(chunks)}")


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Build Chroma database
# --------------------------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_FOLDER
)


print("\n================================")
print("LOREMASTER DATABASE READY")
print("================================")
print(f"Articles indexed: {article_count}")
print(f"Sections indexed: {len(documents)}")
print(f"Chunks indexed:   {len(chunks)}")
