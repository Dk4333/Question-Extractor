from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -------- Config --------

ITINERARY_PATH = Path(__file__).parent / "itinerary.txt"
FALLBACK = "This information is not specified in the provided itinerary."
EMBEDDING_MODEL = "text-embedding-3-small"    # or another OpenAI embedding model
CHAT_MODEL = "gpt-4.1-mini"                   # or gpt-4.1, gpt-4o-mini, etc.
SIM_THRESHOLD = 0.5                           # guardrail for "not in itinerary"


def load_itinerary_text() -> str:
    """Load the fixed itinerary text from disk."""
    if not ITINERARY_PATH.exists():
        raise FileNotFoundError(
            f"Itinerary file not found at {ITINERARY_PATH}. "
            f"Create 'itinerary.txt' next to tripfactory.py."
        )
    return ITINERARY_PATH.read_text(encoding="utf-8")


def build_retriever():
    """Build a retriever over the itinerary text (RAG)."""
    text = load_itinerary_text()

    # Wrap as a single LangChain Document
    docs = [Document(page_content=text)]

    # Split into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    splits = text_splitter.split_documents(docs)

    # Create embeddings + FAISS vector store
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)

    # Return a retriever interface
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


# We'll cache the retriever so it isn't rebuilt on every request
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def build_rag_chain():
    retriever = get_retriever()

    # Prompt: force the model to ONLY use itinerary and use FALLBACK if not present
    template = """
You are a travel assistant.

You must answer user questions ONLY using the CONTEXT below.
If the answer is not clearly present, reply exactly with:
"{fallback}"

When answering, ALWAYS:
- Provide clean, organized bullet points.
- Group similar items together.
- Remove duplicates.
- Do NOT add external knowledge or personal suggestions.
- Keep formatting simple and readable.

CONTEXT:
{context}

QUESTION:
{question}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"fallback": FALLBACK},
    )

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.0,  # deterministic, no creativity
    )

    # Build the RAG chain
    rag_chain = (
        {
            "context": get_retriever(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


_rag_chain = None


def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_chain()
    return _rag_chain


def get_answer(url: str, question: str) -> str:
    """
    Kept the same function name/signature for compatibility with app.py,
    but we IGNORE the 'url' now and always use the fixed itinerary instead.

    This ensures all answers are generated ONLY from itinerary.txt.
    """
    try:
        # Optional: quick retrieval confidence check
        retriever = get_retriever()

        # Embed the question and see if anything similar exists
        # (We can approximate similarity using retriever's search)
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return FALLBACK

        # Simple heuristic: if top doc is extremely short/irrelevant, we could
        # still fall back. For now, rely on LLM + prompt to enforce FALLBACK.
        rag_chain = get_rag_chain()
        answer = rag_chain.invoke(question).strip()

        # Final safety: if the model didn't follow instructions but it's obvious
        # there's no clear answer, you could override with FALLBACK here.
        return answer or FALLBACK

    except Exception as e:
        return f"An error occurred: {e}"