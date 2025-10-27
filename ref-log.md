## External Tools, Libraries, and Docs

- Streamlit — app framework and chat UI  
  - https://docs.streamlit.io/
- LangChain — loaders, text splitters, prompt templates  
  - https://python.langchain.com/
- LangChain OpenAI — ChatOpenAI & OpenAIEmbeddings wrappers  
  - https://python.langchain.com/docs/integrations/llms/openai
- LangChain Chroma — VectorStore integration  
  - https://python.langchain.com/docs/integrations/vectorstores/chroma
- ChromaDB — vector database  
  - https://docs.trychroma.com/
- LangGraph — lightweight graph for retrieve→generate pipeline  
  - https://langchain-ai.github.io/langgraph/
- PDF parsing  
  - `PyPDFLoader` via `langchain_community.document_loaders` (requires `pypdf`)
- Environment management  
  - `python-dotenv` for optional local `.env` usage
- GitHub Codespaces — running, ports, and secrets  
  - https://docs.github.com/en/codespaces

## Class-Provided Materials

- Assignment template & example code (Chunks 1–23) — used as the basis for LLM init, loading, chunking, embeddings, Chroma vector store, retriever setup, and a LangGraph retrieve→generate chain. Adapted into a Streamlit app with multi-file upload, in-memory/disk toggle, and chat UX.

## Design Choices (Brief Rationale)

- In-memory Chroma (default) to avoid read-only SQLite issues on Codespaces; Disk (/tmp) available for persistence across app restarts.  
- RecursiveCharacterTextSplitter with adjustable `chunk_size` and `chunk_overlap` for scalable ingestion.  
- Grounded prompt that refuses to answer outside the provided context; sources and matched chunks shown to verify grounding.  
- History-aware prompting improves follow-up questions without storing private user data.

## GenAI Usage

- Assistant: ChatGPT (GPT-5 Thinking) was used to:
  - Outline implementation steps and map rubric items to features.
  - Troubleshoot Chroma persistence errors in Codespaces and propose an in-memory/disk toggle.
  - Refactor code to read credentials from environment and remove hard-coded keys.
  - Add UX features: history-aware prompting and matched-chunk display.
- Why appropriate: Guidance accelerated development while I retained review and control. All code was reviewed, run, and tested by me inside the class Codespace.

## How to Reproduce
- Environment: GitHub Codespaces using the provided devcontainer.  
- Keys: set `OPENAI_API_KEY` & `OPENAI_BASE_URL=https://api.ai.it.cornell.edu` via Codespaces Secrets.  
- Run:
  ```bash
  streamlit run chat_with_pdf.py --server.address 0.0.0.0 --server.port 8501
  gp url 8501

