

##  Features

-  **LangChain Integration** — Uses LangChain’s document loaders, text splitters, embeddings, and conversational retrieval.  
-  **Conversational Memory** — The bot remembers chat history for better contextual answers.  
-  **Local or API-based Models** — Works with OpenAI, Hugging Face, or local sentence-transformer models.  
-  **Vector Database** — FAISS or Chroma for fast semantic retrieval.  
-  **Frontend Ready** — Optional Gradio or Streamlit interface for a clean chat expe.

---

## Architecture Overview
                                      ┌──────────────────────────┐
                 │       User (Web UI)      │
                 │   - Types a question     │
                 │   - Clicks "Ask"         │
                 └─────────────┬────────────┘
                               │   (POST Request)
                               ▼
                 ┌──────────────────────────┐
                 │     Flask Backend        │
                 │         (app.py)         │
                 │ - Receives question      │
                 │ - Calls RAG engine       │
                 └─────────────┬────────────┘
                               │
                               ▼
             ┌──────────────────────────────────────────┐
             │                 RAG Engine               │
             │               (tripfactory.py)           │
             └──────────────────┬───────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────────┐
        ▼                       ▼                            ▼
┌────────────────┐    ┌────────────────────────┐    ┌───────────────────────────┐
│ Itinerary Data │    │   Retriever (FAISS)    │    │      LLM (OpenAI)         │
│ (itinerary.txt)│    │ - Embeds chunks        │    │ - Uses strict prompt      │
│ - Raw text      │    │ - Vector similarity    │    │ - Answers ONLY using      │
│ - Split chunks  │    │ - Returns top-k chunks │    │   retrieved context       │
└────────────────┘    └────────────────────────┘    └───────────────────────────┘
                                │
                                ▼
                      ┌──────────────────────────┐
                      │     Retrieved Context     │
                      │ (Top-k relevant chunks)   │
                      └─────────────┬────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │     Final Answer (UI)    │
                       │ Rendered back to the user│
                       └──────────────────────────┘
---

##  Tech Stack

- **Python 3.10+**
- **LangChain** — document loading, text processing, and chains  
- **FAISS / Chroma** — vector database  
- **OpenAI / Hugging Face / Local Models** — embeddings and LLMs  
- **BeautifulSoup4 / Requests** — webpage scraping  
- **Gradio / Streamlit** — simple frontend UI  

---

##  Installation

```bash
# 1️ Clone the repository
git clone https://github.com/<your-username>/langchain-url-chatbot.git
cd langchain-url-chatbot

# 2 Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3️ Install dependencies
pip install -r requirements.txt


OPENAI_API_KEY=your_openai_api_key
# or if you’re using Hugging Face embeddings:
HUGGINGFACEHUB_API_TOKEN=your_hf_token
python app.py
streamlit run app.py
# or
python gradio_app.py
Enter URL: https://en.wikipedia.org/wiki/Artificial_intelligence
User: What are the main subfields of AI?
Bot: The major subfields include machine learning, computer vision, natural language processing, robotics, and expert systems.

