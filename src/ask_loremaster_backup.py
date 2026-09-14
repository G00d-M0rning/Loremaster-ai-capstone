import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


DEBUG = True


LOREMASTER_BANNER = r"""
                                  ....
                                .'' .'''
.                             .'   :
\\                          .:    :
 \\                        _:    :       ..----.._
  \\                    .:::.....:::.. .'         ''.
   \\                 .'  #-. .-######'     #        '.
    \\                 '.##'/ ' ################       :
     \\                  #####################         :
      \\               ..##.-.#### .''''###'.._        :
       \\             :--:########:            '.    .' :
        \\..__...--.. :--:#######.'   '.         '.     :
        :     :  : : '':'-:'':'::        .         '.  .'
        '---'''..: :    ':    '..'''.      '.        :'
           \\  :: : :     '      ''''''.     '.      .:
            \\ ::  : :     '            '.      '      :
             \\::   : :           ....' ..:       '     '.
              \\::  : :    .....####\\ .~~.:.             :
               \\':.:.:.:'#########.===. ~ |.'-.   . '''.. :
                \\    .'  ########## \ \ _.' '. '-.       '''.
                :\\  :     ########   \ \      '.  '-.        :
               :  \\'    '   #### :    \ \      :.    '-.      :
              :  .'\\   :'  :     :     \ \       :      '-.    :
             : .'  .\\  '  :      :     :\ \       :        '.   :
             ::   :  \\'  :.      :     : \ \      :          '. :
             ::. :    \\  : :      :    ;  \ \     :           '.:
              : ':    '\\ :  :     :     :  \:\     :        ..'
                 :    ' \\ :        :     ;  \|      :   .'''
                 '.   '  \\:                         :.''
                  .:..... \\:       :            ..''
                 '._____|'.\\......'''''''.:..'''
                            \\

==========================================================================================
                                      L O R E M A S T E R
==========================================================================================

                           AI ARCHIVE OF TOLKIEN'S LEGENDARIUM

                              Knowledge Base .... READY
                              Vector Archive .... READY
                              Local LLM ......... LLAMA 3.2

------------------------------------------------------------------------------------------
                         Speak, traveller. What lore do you seek?
------------------------------------------------------------------------------------------

Type EXIT to leave the archives.
"""


# --------------------------------------------------
# 1. Initialize Loremaster
# --------------------------------------------------

print("Initializing Loremaster...")
print("Loading the archives of Middle-earth...\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# --------------------------------------------------
# 2. Create Loremaster instructions
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are Loremaster, a scholarly assistant specializing in
J.R.R. Tolkien's legendarium.

Your task is to answer the user's question using ONLY the lore
records provided below.

GROUNDING RULES:

1. Base all factual claims on the provided lore records.
2. Do not use outside knowledge.
3. Do not invent facts, characters, events, relationships, dates,
   numbers, durations, or quotations.
4. You may combine information from multiple records when the records
   clearly support the connection.
5. Do not assume that two facts are causally related simply because
   they appear related.
6. When making a reasonable inference rather than stating a direct
   fact, identify it with wording such as "This suggests..."
7. If the records contain conflicting information, do not silently
   choose one version. Briefly acknowledge the difference.

ATTRIBUTION RULES:

Pay careful attention to the ARTICLE and SECTION labels attached to
each lore record.

Never attribute an event, action, relationship, date, or characteristic
from one article to the subject of another article unless the text
explicitly establishes that connection.

For example, information appearing in an Aragorn record describes
Aragorn unless the text explicitly states that the information concerns
another character.

When several records discuss the same subject, prefer the record most
directly concerned with the subject of the question.

DATES AND CHRONOLOGY:

Use dates, numbers, and durations only when they are explicitly
supported by the lore records.

Do not calculate or estimate durations unless the user specifically
asks you to do so.

When describing a sequence of events, preserve the chronology supported
by the records. Do not fill chronological gaps with assumptions.

ANSWERING RULES:

Answer the user's question directly first.

For a simple factual question, give a short and direct factual answer.

For a broader question, combine relevant information from the records
into a clear explanation.

If the records contain only part of the answer, answer the supported
part and briefly explain what information is missing.

Prefer a short accurate answer over a longer speculative answer.

Only say:

"The available lore records do not contain enough information
to answer this question."

when the provided records contain no information that answers the
question.

Do not refuse merely because the records are incomplete.

Before producing your answer, silently check:

- Who performed each action?
- Which lore record supports the claim?
- Is a stated date or number actually present in the records?
- Does the chronology follow the records?
- Am I adding information that the records do not contain?

If a claim fails these checks, omit it.

LORE RECORDS:

{context}

QUESTION:

{question}

LOREMASTER'S ANSWER:
"""
)

# --------------------------------------------------
# 3. Clear startup messages and display Loremaster
# --------------------------------------------------

os.system("cls" if os.name == "nt" else "clear")

print(LOREMASTER_BANNER)


# --------------------------------------------------
# 4. Continuous question loop
# --------------------------------------------------

while True:

    question = input("\nYou > ").strip()

    if not question:
        continue

    if question.lower() in ["exit", "quit", "bye"]:
        print("\nLoremaster > The archives close. Farewell, traveller.\n")
        break

 
     # --------------------------------------------------
    # Entity-aware retrieval
    # --------------------------------------------------

    question_lower = question.lower()

    detected_entities = []

    for alias, article_title in ENTITY_ALIASES.items():
        if alias in question_lower:
            if article_title not in detected_entities:
                detected_entities.append(article_title)

    results = []

    # Search directly inside explicitly mentioned articles
    for article_title in detected_entities:
        entity_results = vector_store.similarity_search(
            question,
            k=4,
            filter={"title": article_title}
        )

        results.extend(entity_results)

    # Add a few general semantic results for context
    semantic_results = vector_store.similarity_search(
        question,
        k=6
    )

    results.extend(semantic_results)

    # Remove duplicate chunks
    unique_results = []
    seen = set()

    for result in results:
        key = (
            result.metadata.get("title"),
            result.metadata.get("section"),
            result.page_content
        )

        if key not in seen:
            seen.add(key)
            unique_results.append(result)

    # Limit final context
    results = unique_results[:10]

    # --------------------------------------------------
    # Build context from retrieved lore
    # --------------------------------------------------

    context_parts = []

    for result in results:
        title = result.metadata.get("title", "Unknown")
        section = result.metadata.get("section", "Unknown")
        source = result.metadata.get("source", "Unknown")

        context_parts.append(
            f"""=== LORE RECORD ===
ARTICLE: {title}
SECTION: {section}
SOURCE: {source}

{result.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------
    # Optional debug display
    # --------------------------------------------------

    if DEBUG:

        print("\n--- DEBUG: RETRIEVED LORE ---")

        for i, result in enumerate(results, start=1):

            print(f"\n[CHUNK {i}]")
            print(
                "TITLE:",
                result.metadata.get("title")
            )
            print(
                "SECTION:",
                result.metadata.get("section")
            )
            print(result.page_content)

    # --------------------------------------------------
    # Build prompt
    # --------------------------------------------------

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # --------------------------------------------------
    # Ask local LLM
    # --------------------------------------------------

    response = llm.invoke(messages)

    # --------------------------------------------------
    # Display answer
    # --------------------------------------------------

    print("\nLoremaster >")
    print(response.content)

    # --------------------------------------------------
    # Display sources
    # --------------------------------------------------

    print("\nSources:")

    sources = set()

    for result in results:
        source = result.metadata.get(
            "source",
            "Unknown source"
        )
        sources.add(source)

    for source in sorted(sources):
        print("-", source)