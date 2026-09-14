import os

from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


DEBUG = False


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

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --------------------------------------------------
# 2. Create Loremaster instructions
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are Loremaster, an expert assistant on J.R.R. Tolkien's legendarium.

Answer the QUESTION using only the LORE RECORDS provided below.

RULES:

1. Answer using ONLY information contained in the LORE RECORDS.
   Do not use outside knowledge of Tolkien's legendarium.

2. Answer the user's question directly and naturally.

3. You may combine information from multiple lore records to create
   a useful answer, summary, biography, timeline, or explanation.

4. For broad questions such as:
   - "Who is..."
   - "Tell me about..."
   - "What is the story of..."
   - "What is the life of..."
   - "Explain..."
   
   synthesize the relevant information in the records into a
   coherent answer. Do not require one passage to contain the
   complete answer.

5. For factual questions, provide the answer when it is supported
   by the records.

6. For questions asking WHY something happened, give a reason only
   when the records support that connection. Do not invent motives
   or causes.

7. For questions asking HOW something happened, describe the
   sequence of events supported by the records.

8. Do not invent dates, events, relationships, motives, dialogue,
   quotations, or sequences of events.

9. Multiple names, aliases, titles, nicknames, or personality
   descriptions do not automatically represent different people.
   If the records identify different names as belonging to the
   same individual, treat them as the same individual.

10. Do not turn interpretations, metaphors, psychological
    descriptions, or speculation into established facts.

11. If the records contain conflicting information, prefer records
    that most directly discuss the subject of the question and
    acknowledge the conflict when necessary.

12. If the records provide only part of an answer, give the useful
    information that IS available and briefly identify what is
    missing.

13. Only say:

    "The available lore records do not contain enough information
    to answer this question."

    when the retrieved records genuinely contain no useful
    information for answering the question.

14. Never refuse to answer merely because the records are
    incomplete. Give the best supported answer possible.

15. Prefer concise answers for simple questions and more detailed
    synthesis for broad questions.

16. For broad biographical questions such as "Who is X?", "Tell me about X",
    or "What is the life of X?", prioritize the character's identity, role,
    major relationships, and major events rather than minor details such as
    name translations.

17. For chronological or biographical questions, organize events in chronological
    order only when their order is established by the records. Do not simply
    present retrieved chunks in retrieval order.

18. Do not display your internal verification process, supporting-statement
    analysis, or reasoning steps. Provide only the final answer.


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

    # --------------------------------------------------
    # STAGE 1: Retrieve candidate chunks with Chroma
    # --------------------------------------------------

    candidates = vector_store.similarity_search(
        question,
        k=20
    )

    # --------------------------------------------------
    # STAGE 2: Rerank candidates
    # --------------------------------------------------

    rerank_pairs = []

    for candidate in candidates:
        title = candidate.metadata.get("title", "Unknown")
        section = candidate.metadata.get("section", "Unknown")

        candidate_text = (
            f"Article: {title}\n"
            f"Section: {section}\n"
            f"{candidate.page_content}"
        )

        rerank_pairs.append(
            [question, candidate_text]
        )

    scores = reranker.predict(rerank_pairs)

    ranked_results = sorted(
        zip(candidates, scores),
        key=lambda item: item[1],
        reverse=True
    )

    results = [
        result
        for result, score in ranked_results[:6]
    ]


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

        for i, (result, score) in enumerate(
            ranked_results[:6],
            start=1
        ):
            print(f"\n[CHUNK {i}]")
            print(f"SCORE: {score:.4f}")
            print("TITLE:", result.metadata.get("title"))
            print("SECTION:", result.metadata.get("section"))
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
