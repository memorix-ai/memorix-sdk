# 📦 Installation Guide

## Quick Install

```bash
pip install memorix-sdk
```

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version recommended
- **API Keys**: For embedding services (OpenAI, Google, etc.)

## Installation Options

### 🚀 Basic Installation

```bash
# Install core package
pip install memorix-sdk
```

### 🔧 Development Installation

```bash
# Clone repository
git clone https://github.com/memorix-ai/memorix-sdk.git
cd memorix-sdk

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 📦 Optional Dependencies

```bash
# Install with specific backends
pip install "memorix-sdk[openai]"      # OpenAI embeddings
pip install "memorix-sdk[gemini]"      # Google Gemini embeddings
pip install "memorix-sdk[faiss]"       # FAISS vector store
pip install "memorix-sdk[qdrant]"      # Qdrant vector store

# Install all optional dependencies
pip install "memorix-sdk[all]"
```

## Environment Setup

### 🔑 API Keys

Set up your API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# Cohere (future)
export COHERE_API_KEY="your-cohere-api-key"
```

### 📁 Configuration File

Create a `memorix.yaml` configuration file:

```yaml
vector_store:
  type: faiss
  index_path: ./memorix_index
  dimension: 1536

embedder:
  type: openai
  model: text-embedding-ada-002
  api_key: ${OPENAI_API_KEY}

metadata_store:
  type: sqlite
  database_path: ./memorix_metadata.db

settings:
  max_memories: 10000
  similarity_threshold: 0.7
```

## Platform-Specific Instructions

### 🐧 Linux (Ubuntu/Debian)

```bash
# Update package manager
sudo apt update

# Install Python dependencies
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv memorix-env
source memorix-env/bin/activate

# Install Memorix SDK
pip install memorix-sdk
```

### 🍎 macOS

```bash
# Using Homebrew
brew install python3

# Create virtual environment
python3 -m venv memorix-env
source memorix-env/bin/activate

# Install Memorix SDK
pip install memorix-sdk
```

### 🪟 Windows

```bash
# Using PowerShell
python -m venv memorix-env
memorix-env\Scripts\Activate.ps1

# Install Memorix SDK
pip install memorix-sdk
```

### 🐳 Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Memorix SDK
RUN pip install memorix-sdk

# Copy your application
COPY . .

CMD ["python", "your_app.py"]
```

## Backend-Specific Setup

### 🗄️ Vector Stores

#### FAISS
```bash
# CPU version (default)
pip install faiss-cpu

# GPU version (optional)
pip install faiss-gpu
```

#### Qdrant
```bash
# Install Qdrant client
pip install qdrant-client

# Run Qdrant server (optional)
docker run -p 6333:6333 qdrant/qdrant
```

### 🤖 Embedding Models

#### OpenAI
```bash
# Install OpenAI client
pip install openai

# Set API key
export OPENAI_API_KEY="your-api-key"
```

#### Google Gemini
```bash
# Install Google Generative AI
pip install google-generativeai

# Set API key
export GOOGLE_API_KEY="your-api-key"
```

#### Sentence Transformers
```bash
# Install sentence transformers
pip install sentence-transformers torch

# No API key required - runs locally
```

### 📊 Metadata Stores

#### SQLite
```bash
# Built into Python - no additional installation needed
```

#### PostgreSQL (future)
```bash
# Install PostgreSQL adapter
pip install psycopg2-binary
```

## Verification

### 🧪 Test Installation

```python
# Test basic functionality
from memorix import MemoryAPI, Config

# Create configuration
config = Config()

# Initialize memory API
memory = MemoryAPI(config)

# Test store operation
memory_id = memory.store("Hello, Memorix!")
print(f"Stored memory with ID: {memory_id}")

# Test retrieve operation
results = memory.retrieve("Hello")
print(f"Found {len(results)} memories")
```

### 🔍 Check Installation

```bash
# Check installed version
pip show memorix-sdk

# Run tests (if installed in dev mode)
python -m pytest tests/ -v

# Check available backends
python -c "from memorix import MemoryAPI, Config; print('Installation successful!')"
```

## Troubleshooting

### ❌ Common Issues

#### Import Errors
```bash
# Ensure you're in the correct virtual environment
source memorix-env/bin/activate  # Linux/macOS
memorix-env\Scripts\Activate.ps1  # Windows
```

#### API Key Issues
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
python -c "import openai; openai.api_key='$OPENAI_API_KEY'; print('API key valid')"
```

#### Permission Errors
```bash
# Fix permission issues
sudo chown -R $USER:$USER ~/.cache/pip
```

#### Memory Issues
```bash
# Increase swap space (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 🔧 Performance Tuning

#### FAISS Optimization
```python
# Use GPU if available
config.set('vector_store.use_gpu', True)

# Optimize index type
config.set('vector_store.index_type', 'IVF100,SQ8')
```

#### Batch Processing
```python
# Use batch operations for large datasets
embeddings = embedder.embed_batch(texts)
```

## Next Steps

1. **Read the [Quick Start Guide](../README.md#quick-start)**
2. **Explore [Examples](../examples/)**
3. **Check [Configuration Options](../memorix.yaml)**
4. **Join the [Community](https://github.com/memorix-ai/memorix-sdk/discussions)**

## Support

- 📚 **Documentation**: [docs.memorix.ai](https://docs.memorix.ai)
- 🐛 **Issues**: [GitHub Issues](https://github.com/memorix-ai/memorix-sdk/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/memorix-ai/memorix-sdk/discussions)
- 📧 **Email**: support@memorix.ai 