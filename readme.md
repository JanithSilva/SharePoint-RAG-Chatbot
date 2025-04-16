# SharePoint RAG Chatbot

## Overview

The SharePoint RAG Chatbot is a system designed to process documents stored in SharePoint using AI services and store the results in vector and graph databases. It also provides a chatbot for interacting with the processed data.

## Features

- Monitors SharePoint for new documents.
- Processes documents using Azure Document Intelligence.
- Stores processed data in Pinecone (vector database) and a graph database.
- Provides a chatbot interface for querying the data.

## Prerequisites

- Python 3.8 or higher
- SharePoint access credentials
- Azure Form Recognizer access credentials
- Pinecone API access credentials
- Neo4j database access credentials
- Install required Python packages (see `requirements.txt`).

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/JanithSilva/SharePoint-RAG-Chatbot.git
cd SharePoint-RAG-Chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# SharePoint settings
SHAREPOINT_SITE_URL=
SHAREPOINT_USERNAME=
SHAREPOINT_PASSWORD=
SHAREPOINT_LIBRARY_NAME=

# Azure Document Intelligence settings
AZURE_DOCUMENT_INTEL_KEY=
AZURE_DOCUMENT_INTEL_ENDPOINT=

# Pinecone settings
PINECONE_API_KEY=

# Neo4j settings
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=

# Using Azure OpenAI Embeddings
EMBEDDING_MODEL_ENDPOINT=
EMBEDDING_MODEL_KEY=
EMBEDDING_MODEL_API_VERSION=
EMBEDDING_MODEL_DEPLOYMENT_NAME=
EMBEDDING_MODEL_CHUNK_SIZE=

# LLM API settings (Azure OpenAI API)
LLM_API_VERSION=
LLM_API_ENDPOINT=
LLM_API_KEY=
LLM_DEPLOYMENT_NAME=
```

### 5. MCP-Server Setup

- Clone the SharePoint-RAG-Chatbot Repository:

```bash
git clone https://github.com/JanithSilva/MCP-Server.git
cd MCP-Server
```

- Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

- Set up environment variables:

```env
SHAREPOINT_URL=
SHAREPOINT_USERNAME=
SHAREPOINT_PASSWORD=
SHAREPOINT_LIBRARY_NAME=

NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=

EMBEDDING_MODEL_ENDPOINT=
EMBEDDING_MODEL_KEY=
EMBEDDING_MODEL_API_VERSION=
EMBEDDING_MODEL_DEPLOYMENT_NAME=
EMBEDDING_MODEL_CHUNK_SIZE=2048
```

- Start the local MCP server:

```bash
python mcp_server.py
```

- Add MCP server details to the MCP Client in the RAG Chatbot.

## Usage

Ensure the MCP server is running before starting the LangGraph development server.

To start the local development server, run the following command:

```bash
langgraph dev
```

You should see the following output:

- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs

Open your browser and navigate to:

```plaintext
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## Project Structure

```plaintext
├── src/
│   ├── app.py              # FastAPI application
│   ├── auth.py             # Authentication utilities
│   ├── settings.py         # Configuration settings
│   ├── agents/             # LangGraph agents
│   ├── services/           # Business logic services
│   └── mcp_server/         # MCP server components
├── data/                   # Data storage
├── tests/                  # Test files
├── requirements.txt        # Python dependencies└
── .env                    # Environment variables
```
