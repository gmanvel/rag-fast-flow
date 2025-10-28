# Fast Flow RAG Comparison Application

A Streamlit-based application that demonstrates the difference between pure LLM responses and RAG (Retrieval-Augmented Generation) enhanced responses for questions about Fast Flow methodologies including Wardley Mapping, Domain-Driven Design (DDD), and Team Topologies.

## Overview

This application allows users to ask questions and see two responses side-by-side:

1. **Pure LLM Response**: Direct query to the Mistral model without additional context
2. **RAG-Enhanced Response**: Query enriched with relevant sections retrieved from a vector database (Qdrant)

This comparison helps illustrate how RAG improves response accuracy and relevance by grounding the LLM's answers in specific documentation.

## Architecture

```
┌─────────────┐
│    User     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│   Streamlit UI (Docker Container)       │
│   - User input                          │
│   - Side-by-side response display       │
└────────┬───────────────────────┬────────┘
         │                       │
         │ Pure Query            │ RAG Query
         │                       │
         ▼                       ▼
    ┌────────┐            ┌──────────────┐
    │        │            │   Qdrant     │
    │        │            │  (Docker)    │
    │        │            │  Vector DB   │
    │        │            └──────┬───────┘
    │        │                   │
    │        │                   │ Context
    │        │                   │ Retrieval
    │ Mistral│◄──────────────────┘
    │  LLM   │
    │(Ollama)│      Query + Context
    │ Host   │
    └────┬───┘
         │
         ▼
    Responses
```

### Component Description

- **Streamlit App**: Web UI running in Docker, orchestrates user interactions
- **Qdrant**: Vector database running in Docker, stores document embeddings and enables semantic search
- **Ollama/Mistral**: LLM running on host machine, generates responses
- **Embedding Model**: nomic-embed-text model via Ollama, converts text to vectors

### How It Works

1. **Pure LLM Path**:
   - User question → Mistral LLM → Response

2. **RAG Path**:
   - User question → Embedding generation (nomic-embed-text)
   - Query vector → Qdrant search → Top-k relevant chunks
   - User question + Retrieved context → Mistral LLM → Enhanced response

3. **Data Population (One-time)**:
   - Load `fast_flow_extracted.json` → Filter sections (non-empty titles, exclude "Summary")
   - For each section → **SemanticSplitterNodeParser** splits into coherent chunks
   - For each chunk → Generate embedding (768-dim vector)
   - Insert all chunks into Qdrant with parent section metadata

### Semantic Chunking

The application uses **SemanticSplitterNodeParser** from LlamaIndex to intelligently split document sections into smaller, semantically coherent chunks. This approach:

- **Preserves meaning**: Chunks are split based on semantic similarity, not arbitrary character counts
- **Optimizes retrieval**: Smaller chunks allow for more precise matching to user queries
- **Improves context quality**: Retrieved chunks are more focused and relevant
- **Configurable**: Uses `breakpoint_percentile_threshold=70` to determine split points

Each chunk maintains a reference to its parent section's title, allowing users to see which part of the documentation was retrieved.

## Prerequisites

Before running this application, ensure you have the following installed and configured:

### Required Software

- **Docker** (version 20.10 or later)
- **Docker Compose** (version 2.0 or later)
- **Ollama** running locally on your host machine

### Ollama Setup

You need to have Ollama installed and running with the required models:

1. **Install Ollama**: Follow instructions at [ollama.ai](https://ollama.ai)

2. **Pull required models**:
   ```bash
   # Pull Mistral LLM model
   ollama pull mistral

   # Pull embedding model
   ollama pull nomic-embed-text
   ```

3. **Verify Ollama is running**:
   ```bash
   # Check that Ollama is accessible
   curl http://localhost:11434/api/tags
   ```

   You should see a JSON response listing your available models.

### Qdrant Data Population

**Good News!** The application now includes a built-in feature to populate Qdrant directly from the UI. You don't need to run any notebook cells manually.

The application will automatically detect if Qdrant is empty and show a "Populate Qdrant" button in the sidebar. When you click it, the app will:

1. Process the Fast Flow documentation from `data/fast_flow_extracted.json`
2. Split sections into semantic chunks using AI-powered semantic splitting
3. Generate embeddings for each chunk using the nomic-embed-text model
4. Insert all chunks into Qdrant

**Note**: The population process may take 2-5 minutes depending on your system. The app will show progress updates and automatically refresh when complete.

## Quick Start

### 1. Clone and Navigate

```bash
cd /path/to/rag-fast-flow
```

### 2. Start the Application

```bash
docker-compose up -d
```

This command will:
- Start Qdrant vector database (ports 6333, 6334)
- Build and start the Streamlit application (port 8501)

### 3. Verify Services

Check that all services are running:

```bash
docker-compose ps
```

You should see both `rag-qdrant` and `rag-streamlit` containers running.

### 4. Access the Application

Open your browser and navigate to:

```
http://localhost:8501
```

### 5. Populate Qdrant Database (First Time Only)

When you first access the application, the sidebar will show that Qdrant is empty and display a "🚀 Populate Qdrant" button.

**Click the button** to populate the database with Fast Flow documentation. This process:
- Reads and processes the Fast Flow book content
- Creates semantic chunks from each section
- Generates embeddings for ~300-500 chunks
- Takes approximately 2-5 minutes

The page will automatically refresh when complete.

### 6. System Status

The application sidebar shows:
- ✅ Qdrant connection status
- 📊 Number of chunks loaded (should show ~300-500 after population)
- 🔢 Vector dimensions (768 for nomic-embed-text)

## Usage

### Asking Questions

1. **Enter a question** in the text area about Fast Flow methodologies, for example:
   - "What is Wardley Mapping?"
   - "How does Domain-Driven Design help with software architecture?"
   - "Explain Team Topologies and their benefits"
   - "How do bounded contexts relate to team organization?"

2. **Click "🔍 Get Answers"** to see both responses

3. **Compare the responses** side-by-side:
   - **Left column** (💭 Pure LLM Response): Direct answer from Mistral without additional context
   - **Right column** (🎯 RAG-Enhanced Response): Answer enriched with relevant documentation
   - Expand "📚 Retrieved Context" to see what specific sections were retrieved from the knowledge base

4. **Click "🗑️ Clear"** to reset and ask another question

### Repopulating the Database

If you need to repopulate Qdrant (e.g., after updates to the source data):
1. The app currently recreates the collection each time you click "Populate Qdrant"
2. Any existing data will be replaced
3. The process takes 2-5 minutes to complete

## Configuration

The application can be configured via environment variables in [docker-compose.yml](docker-compose.yml):

### Streamlit Service Environment Variables

```yaml
environment:
  - QDRANT_HOST=qdrant           # Qdrant host (default: localhost)
  - QDRANT_PORT=6333             # Qdrant port (default: 6333)
  - OLLAMA_BASE_URL=http://host.docker.internal:11434  # Ollama API URL
```

### Modifying Configuration

To change settings:

1. Edit [docker-compose.yml](docker-compose.yml)
2. Restart services:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

## Development

### Running Locally (without Docker)

If you want to run the Streamlit app locally for development:

```bash
# Install uv if you haven't already
pip install uv

# Install dependencies
uv pip install -e .

# Set environment variables for local Qdrant
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export OLLAMA_BASE_URL=http://localhost:11434

# Run Streamlit
streamlit run app.py
```

### Project Structure

```
rag-fast-flow/
├── app.py                  # Main Streamlit application
├── llm_service.py          # LLM interaction service
├── rag_service.py          # RAG retrieval service
├── pyproject.toml          # Python dependencies (uv)
├── Dockerfile              # Streamlit app container
├── docker-compose.yml      # Docker orchestration
├── .dockerignore          # Docker build exclusions
├── README.md              # This file
├── raw_poc.ipynb          # Original POC notebook
└── data/
    ├── fast_flow_extracted.json
    └── sections_with_embeddings.json
```

## Troubleshooting

### Qdrant Connection Failed

**Symptom**: Sidebar shows "❌ Qdrant connection failed"

**Solutions**:
- Ensure Qdrant container is running: `docker-compose ps`
- Check Qdrant logs: `docker-compose logs qdrant`
- Verify port 6333 is not in use by another service

### Collection Not Found or Empty

**Symptom**: "⚠️ Qdrant connected but collection not found" or "⚠️ Qdrant collection is empty"

**Solution**:
- Click the "🚀 Populate Qdrant" button in the sidebar
- Wait 2-5 minutes for the population process to complete
- The page will automatically refresh when done

**If population fails**:
- Ensure `data/fast_flow_extracted.json` exists in your project directory
- Check Ollama is running and accessible
- Review Streamlit logs: `docker-compose logs streamlit`

### Ollama Connection Error

**Symptom**: "Error communicating with LLM"

**Solutions**:
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Ensure models are pulled: `ollama list`
- Check Docker can reach host: Test with `docker run --rm curlimages/curl curl http://host.docker.internal:11434/api/tags`

### Slow Response Times

**Causes**:
- First query after startup takes longer (model loading)
- Large context retrieved from Qdrant
- Host machine resource constraints

**Solutions**:
- Wait for first query to complete (subsequent queries will be faster)
- Reduce `top_k` parameter in [rag_service.py](rag_service.py:20) (default: 3)
- Ensure sufficient RAM for Ollama models

### Database Population Takes Too Long

**Symptom**: "Populate Qdrant" process takes more than 10 minutes

**Causes**:
- Slow embedding generation (depends on CPU/GPU)
- Network issues connecting to Ollama
- Large number of sections being processed

**Solutions**:
- Ensure Ollama is running locally (not over network)
- Check system resources: `docker stats`
- Monitor progress in Streamlit logs: `docker-compose logs -f streamlit`
- If it hangs, restart: `docker-compose restart streamlit` and try again

**Note**: Normal processing time is 2-5 minutes for ~300-500 chunks

### Docker Build Fails

**Symptom**: `docker-compose up` fails during build

**Solutions**:
- Ensure you have internet connectivity (downloads Python packages)
- Clear Docker cache: `docker-compose build --no-cache`
- Check Docker has sufficient disk space

## Stopping the Application

To stop all services:

```bash
docker-compose down
```

To stop and remove volumes (deletes Qdrant data):

```bash
docker-compose down -v
```

## Future Enhancements

Potential improvements for this application:

- [x] ~~Add data population UI/script within the application~~ ✅ Completed!
- [ ] Support for multiple document collections
- [ ] Adjustable retrieval parameters (top_k, similarity threshold) via UI
- [ ] Progress bar for database population showing chunk processing
- [ ] Response quality metrics and scoring
- [ ] Export comparison results to PDF/Markdown
- [ ] Support for different LLM models (configurable in UI)
- [ ] Chat history and conversation context
- [ ] Upload custom documents for RAG processing

## License

This project is for educational and demonstration purposes.

## Contributing

Feel free to submit issues or pull requests for improvements!
