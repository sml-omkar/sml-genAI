# AI-Bot — Enterprise Teams Chatbot with RAG Pipeline

A production-ready Microsoft Teams chatbot that answers employee questions from uploaded PDF policy documents. Built with Python FastAPI, Ollama (llama3.2:1b), ChromaDB, and PostgreSQL.

## Features

- **RAG Pipeline** — Upload PDFs, extract text, chunk, embed, and search with hybrid vector + keyword matching
- **Anti-Hallucination** — Bot only answers from uploaded documents, refuses questions outside scope
- **Conversation Memory** — Multi-turn conversations with 24-hour TTL, context carryover between messages
- **Page-Aware Chunking** — Chunks are tied to specific PDF pages for accurate citation
- **Admin Portal** — Windows XP-style UI for managing folders, documents, users, and testing chat
- **RBAC** — Role-based access control with department filtering (HR, IT, Finance)
- **Teams Integration** — Bot Framework SDK for Microsoft Teams deployment

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.9+, FastAPI |
| LLM | Ollama + llama3.2:1b |
| Embeddings | Ollama + nomic-embed-text (768-dim) |
| Vector DB | ChromaDB (persistent) |
| Database | PostgreSQL 16 |
| Auth | JWT tokens |
| Bot | Bot Framework Python SDK |
| PDF Extraction | PyMuPDF4LLM, pdfplumber, Tesseract OCR |

## Ubuntu Setup (Step by Step)

### 1. System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git build-essential software-properties-common

# Install Python 3.9+
sudo apt install -y python3.9 python3.9-venv python3.9-dev python3-pip

# Install PostgreSQL 16
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt update
sudo apt install -y postgresql-16 postgresql-client-16

# Install Tesseract OCR (for scanned PDFs)
sudo apt install -y tesseract-ocr tesseract-ocr-eng

# Install system libraries for PyMuPDF
sudo apt install -y libmupdf-dev libfreetype6-dev libharfbuzz-dev
```

### 2. Ollama (LLM Runtime)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull required models (download ~1.5GB total)
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# Verify models are installed
ollama list
```

### 3. PostgreSQL Database

```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql -c "CREATE USER aibot_user WITH PASSWORD 'aibot_pass';"
sudo -u postgres psql -c "CREATE DATABASE aibot_db OWNER aibot_user;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE aibot_db TO aibot_user;"

# Verify connection
psql -h localhost -U aibot_user -d aibot_db -c "SELECT 1;"
```

### 4. Clone and Setup Application

```bash
# Clone repository
git clone https://github.com/your-org/sml-genAI.git
cd sml-genAI

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Environment Configuration

Create `.env` file in the project root:

```bash
cat > .env << 'EOF'
# Application
APP_NAME=AI-Bot
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false
SECRET_KEY=your-secret-key-change-this

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=aibot_user
POSTGRES_PASSWORD=aibot_pass
POSTGRES_DB=aibot_db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b

# Embeddings
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
EMBEDDING_DIMENSIONS=768

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=policy_documents

# JWT Auth
JWT_SECRET_KEY=your-jwt-secret-change-this
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=480

# Upload
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=50

# RAG
CHUNK_SIZE=384
CHUNK_OVERLAP=100
TOP_K_RESULTS=5
LLM_TEMPERATURE=0.1

# Memory (conversation history)
CONVERSATION_TTL_HOURS=24
MEMORY_MAX_MESSAGES=10

# Microsoft Teams Bot (optional — for Teams integration)
MicrosoftAppType=MultiTenant
MicrosoftAppId=
MicrosoftAppPassword=
MicrosoftAppTenantId=
EOF
```

### 6. Start the Application

```bash
# Make sure Ollama is running
ollama serve &

# Make sure PostgreSQL is running
sudo systemctl start postgresql

# Start the application
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Access the Application

Open your browser and go to:

```
http://localhost:8000/admin
```

**Default admin credentials:**
- Email: `admin@company.com`
- Password: `admin123`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Login and get JWT token |
| GET | `/api/admin/users` | List all users |
| POST | `/api/admin/users` | Create a new user |
| GET | `/api/folders` | List folders |
| POST | `/api/folders` | Create a folder |
| POST | `/api/documents/upload` | Upload a PDF |
| GET | `/api/documents/{id}/status` | Check processing status |
| POST | `/api/chat` | Chat with the bot |
| POST | `/api/messages` | Teams Bot Framework endpoint |

## Project Structure

```
sml-genAI/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Environment settings
│   ├── database.py          # PostgreSQL connection
│   ├── models/
│   │   ├── user.py          # User, Role, Department models
│   │   ├── folder.py        # Folder model
│   │   ├── document.py      # Document + processing status
│   │   └── conversation.py  # Conversation + message history
│   ├── memory/
│   │   └── service.py       # Memory service (24h TTL)
│   ├── rag/
│   │   ├── pdf_extractor.py # PDF → text (3-tier fallback)
│   │   ├── chunker.py       # Text → chunks (page-aware)
│   │   ├── embeddings.py    # Ollama embeddings
│   │   ├── vectorstore.py   # ChromaDB + hybrid search
│   │   └── chain.py         # RAG query pipeline
│   ├── bot/
│   │   ├── bot_handler.py   # Teams bot handler
│   │   ├── adapter.py       # Bot Framework adapter
│   │   └── card_builder.py  # Adaptive Card templates
│   ├── admin/               # Admin API routes
│   ├── auth/                # JWT authentication
│   └── rbac/                # Role-based access control
├── templates/               # Jinja2 HTML templates
├── static/                  # CSS, JS, images
├── data/
│   ├── uploads/             # Uploaded PDFs
│   └── chroma/              # ChromaDB persistence
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md                # This file
```

## How the RAG Pipeline Works

1. **Upload** — PDF is uploaded via admin portal or API
2. **Extract** — Text is extracted using PyMuPDF4LLM → pdfplumber → Tesseract OCR (3-tier fallback)
3. **Chunk** — Text is split by page markers first, then chunked within each page (384 chars, 100 overlap)
4. **Embed** — Each chunk is embedded using nomic-embed-text via Ollama API
5. **Store** — Embeddings are stored in ChromaDB with metadata (document, department, page number)
6. **Query** — User question is embedded, hybrid search finds relevant chunks (vector + keyword matching)
7. **Generate** — LLM generates answer using retrieved chunks + conversation history
8. **Verify** — Post-processing checks for echo detection, hallucination, and "I don't know" retries

## Conversation Memory

- Each conversation has a 24-hour TTL (auto-expires)
- Last 10 messages are included as context for follow-up questions
- Users can ask "What about X?" and the bot understands the reference
- Conversations are stored in PostgreSQL (conversations + messages tables)

## PDF Processing Details

| Stage | Tool | Purpose |
|-------|------|---------|
| Text extraction | PyMuPDF4LLM | Clean Markdown with page markers |
| Table extraction | pdfplumber | Complex tables as Markdown |
| OCR fallback | Tesseract | Scanned/image-based PDFs |
| Chunking | RecursiveCharacterTextSplitter | 384 chars, 100 overlap, page-aware |
| Embedding | nomic-embed-text | 768-dimension vectors |
| Storage | ChromaDB | Persistent vector store |

## Troubleshooting

### Ollama not responding
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve &
```

### PostgreSQL connection refused
```bash
# Check status
sudo systemctl status postgresql

# Restart
sudo systemctl restart postgresql

# Check if database exists
psql -h localhost -U aibot_user -d aibot_db -c "\dt"
```

### Documents stuck in processing
```bash
# Check logs
tail -f /tmp/aibot.log

# The app auto-retries stuck documents on restart
# Or manually reset status in database:
psql -h localhost -U aibot_user -d aibot_db \
  -c "UPDATE documents SET status='failed' WHERE status NOT IN ('ready','failed');"
```

### Port 8000 already in use
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:1b` | LLM model for answer generation |
| `CHUNK_SIZE` | `384` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Character overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks to retrieve |
| `CONVERSATION_TTL_HOURS` | `24` | Hours before conversation expires |
| `MEMORY_MAX_MESSAGES` | `10` | Max messages in conversation history |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF upload size |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `480` | JWT token lifetime (8 hours) |

## License

Internal use only — Sanghvi Movers Limited.
