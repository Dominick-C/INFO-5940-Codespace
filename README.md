-------------------------------------------------------------------------------------------------------------------------------------------------
Running the app (GitHub Codespaces):
Open your fork in GitHub Codespaces.
Paste your API key into the dev container:
Open: .devcontainer/devcontainer.json
Find the "remoteEnv" block and set these values (replace with your own key):
"OPENAI_API_KEY": "sk-PASTE_YOUR_KEY_HERE",
"API_KEY": "sk-PASTE_YOUR_KEY_HERE",
"OPENAI_BASE_URL": "https://api.ai.it.cornell.edu",
"BASE_URL": "https://api.ai.it.cornell.edu/",
"TZ": "America/New_York"
Save the file.
Rebuild the container so the environment is applied:
Command Palette -> Codespaces: Rebuild Container
Verify the environment in the terminal:
echo "$OPENAI_API_KEY" | wc -c (should print a number > 0)
echo "$OPENAI_BASE_URL" (should print https://api.ai.it.cornell.edu)
Run the Streamlit app from the repo root:
streamlit run chat_with_pdf.py --server.address 0.0.0.0 --server.port 8501
Then open the forwarded URL:
gp url 8501
If gp is not available:
echo "https://${CODESPACE_NAME}-8501.app.github.dev"
Using the app:
Upload .txt and/or .pdf files (multiple allowed).
Click “Build / Update Index”.
Ask questions in the chat box.
Use “Reset chat” to clear messages, and “Clear index” to rotate to a fresh collection (prevents read-only DB errors).
Note: Do not commit your personal API key to the repository. Remove or blank the key in .devcontainer/devcontainer.json before pushing.'

-------------------------------------------------------------------------------------------------------------------------------------------------
Features overview 
Multi-file ingestion: supports .txt and .pdf; 

PDFs are parsed page-by-page before chunking. 

Chunking: uses RecursiveCharacterTextSplitter; chunk size and overlap are adjustable in the sidebar. 

Vector database: Chroma with OpenAI embeddings; toggle between In-memory (recommended on Codespaces) and Disk (/tmp per session) to avoid read-only database issues. 
Retrieval: similarity search with configurable top-k. 

Generation: answers are produced by a language model using only retrieved context; if the answer isn’t in the documents, the app says “I don’t know from the provided documents.”

Chat interface: multi-turn conversation with session history. 

Transparency: “Sources” (filename, page, chunk id, score) and “Show matched chunks” (actual retrieved snippets) under each answer. 

No secrets in code: the application reads API keys and base URL from environment variables. 

-------------------------------------------------------------------------------------------------------------------------------------------------
Configuration changes (what was modified and why)
chat_with_pdf.py: implements the complete RAG pipeline and the Streamlit chat UI; adds a toggle for In-memory vs Disk (/tmp) Chroma to prevent read-only errors; includes history-aware prompting and matched-chunk display; reads API credentials from environment variables (API_KEY or OPENAI_API_KEY, and OPENAI_BASE_URL).

.devcontainer/devcontainer.json: uses remoteEnv names that the app reads (OPENAI_API_KEY, API_KEY, OPENAI_BASE_URL). Do not commit real keys; prefer Codespaces Secrets. Optional quality-of-life: forward port 8501 and auto-open in browser. 

requirements.txt: ensure these packages are present if not already in the class template: streamlit, langchain, langchain-openai, langchain-chroma, chromadb, langgraph, langchain-text-splitters, langchain-community, pypdf, python-dotenv. 
-------------------------------------------------------------------------------------------------------------------------------------------------
Troubleshooting (not requierd just thought this could be helpful)
“OPENAI_API_KEY / API_KEY is not set” banner: rebuild the container so remoteEnv or Codespaces Secrets are applied; confirm with echo commands above. 

No browser prompt: always bind to 0.0.0.0 and a fixed port (e.g., 8501). Use gp url 8501 or the Ports panel (open the globe icon). 

Read-only database errors: use “In-memory (recommended)” mode, or Disk mode under /tmp (the app creates a fresh subfolder per session)

PDFs not parsing: ensure pypdf is installed; try another PDF to rule out file corruption. 

Local (non-Codespaces) runs: export OPENAI_API_KEY and OPENAI_BASE_URL in your shell, or use a local .env that is gitignored.
