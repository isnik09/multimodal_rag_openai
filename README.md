# MULTIMODAL RAG IMPLIMENTATION with OPENAI as LLM

### Requirements 

| Package              | Purpose |
|----------------------|---------|
| **streamlit**        | Builds the web-based interactive UI. |
| **openai**           | Provides access to OpenAI’s LLMs (e.g., GPT-4). |
| **faiss-cpu**        | Efficient vector store for similarity search on embeddings. |
| **sentence-transformers** | Generates embeddings from text using transformer models like `all-MiniLM`. |
| **torch**            | PyTorch deep learning backend for sentence-transformers and open-clip. |
| **Pillow**           | Loads and preprocesses image files (JPG, PNG, etc.). |
| **ftfy**             | Automatically fixes broken Unicode text (e.g., from PDFs). |
| **regex**            | Advanced regular expression support (more powerful than Python’s built-in `re`). |
| **tqdm**             | Displays progress bars during embedding/indexing. |
| **open-clip-torch**  | Generates image embeddings using OpenCLIP models (alternative to `clip-by-openai`). |
| **python-dotenv**    | Loads environment variables like `OPENAI_API_KEY` from a `.env` file. |
| **python-docx**      | Extracts and processes content from `.docx` Word files. |
| **pymupdf** (`fitz`) | Parses and extracts text/images from PDF files quickly and accurately. |

