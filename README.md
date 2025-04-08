# RAG for U.S. Congressional Bills

## Overview

This project develops a Retrieval-Augmented Generation (RAG) system that uses a LLM to provide concise summaries of U.S. congressional bills from the 115th to 119th Congress. The system features a web interface built with Streamlit and with pre-built Docker image which can be deployed on Google Cloud Platform.

## Key Features

- **Data Processing:**

  - Extract bill texts from XML files.
  - Chunk texts into 512-word segments.
  - Fine-tune a small FLAN-t5 for topic tagging.

- **Retrieval & Generation:**
  - Experiment with FAISS and Marqo for vector-based retrieval.
  - Experiment with different LLMs to generate answers which match the information retrieved for the vector database.

## Data Sources

- Primary legislative documents from [GovInfo](https://www.govinfo.gov/app/collection/BILLS).
- Additional references from [GovTrack](https://www.govtrack.us/congress/bills/)

## Deployment

- **Web Interface:** Built with Streamlit.
- **Docker:** A pre-built Docker image is provided for easy deployment.
- **Hosting:** Deployed on Google Cloud Platform.

## How to Run with GPUs

1. Procure GPU GCP machine with appropriate boot disk and image
2. Clone this repository
3. Run ./docker-startup build
4. Run ./docker-startup deploy-gpu
