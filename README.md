# RAG for U.S. Congressional Bills

## Overview

This project develops a Retrieval-Augmented Generation (RAG) system that uses a LLM to provide concise summaries of U.S. congressional bills from the 115th to 119th Congress. The system features a web interface built with Streamlit and with pre-built Docker image which can be deployed on Google Cloud Platform.

## Key Features

- **Data Processing:**

  - Extract bill texts from XML files.
  - Chunk texts into 512-word segments with overlap using LangChain.
  - Fine-tune a small FLAN-t5 for topic tagging.

- **Retrieval & Generation:**
  - Experiment with FAISS and all-MiniLM-L6-v2 (or Marqo in POC) for embedded vector-based retrieval.
  - Experiment with different LLMs to generate answers which match the information retrieved for the vector database.
    - deepseek/deepseek-r1
    - google/gemini-2.0-flash
    - qwen/qwq-32b
    - deepseek/deepseek-chat-v3-0324
    - Meta-Llama-3.1-8B-Instruct
    - Qwen/Qwen2.5-7b-instruct-1m
    - Qwen/Qwen2.5-1.5B-Instruct
    - mistral-7b-instruct-v0.1

## Data Sources and Verification

- Primary legislative documents from [GovInfo](https://www.govinfo.gov/app/collection/BILLS).
- For output verification, head to [GovInfo](https://www.govinfo.gov/) and press "Search." Then enter the value inside the parentheses (id...) from the generated output, but remove the beginning "id" character. 
- Additional references from [GovTrack](https://www.govtrack.us/congress/bills/)

## Deployment

- **Web Interface:** Built with Streamlit.
- **Docker:** Built Docker image with provided [docker file](serving-rag-gemini/Dockerfile) in the repository.
- **Hosting:** Deployed on Google Cloud Platform.

## How to Run POC

1. Marqo requires Docker. To install Docker go to Docker Docs and install for your operating system.Once Docker is installed, you can use it to run Marqo. First, open the Docker application and then head to your terminal and enter the following:

```bash
docker pull marqoai/marqo:latest
docker rm -f marqo
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
```

2. Generate index using Marqo:

```bash
git clone https://github.com/sruxll/rag_us_congressional_bills.git
cd archive/POC
```

```python
python generate_index_marqo.py
```

3. Login Huggingface:

```bash
huggingface-cli login <YOUR-HUGGINGFACE-ACCESS-TOKEN>
```

4. Start streamlit app:

```python
streamlit run streamlit_app.py
```

## How to Run the Streamlit App on GCP 🚀

1. Get your free Gemini API key

- go to [Google AI Studio](https://aistudio.google.com/)
- create a new api key in a new project or an existing one in GCP, which is `<YOUR_GEMINI_API_KEY>`, and save it.

2. Procure GCP machine with appropriate boot disk, image and metadata setup using the following recommended settings:

- Machine configuration:
  - type: select `N2` with `n2-standard-2 (2 vCPU, 1 core, 8 GB memory)`
- OS and storage: change the public images to
  - operating system: `Deep Learning on Linux`
  - version: `Deep Learning VM M129`
  - size: `100GB`
- Networking:
  - Allow HTTP traffic
  - Allow HTTPS traffic
- Advanced: add gemini api key into Metadata section:
  - key: `GEMINI_API_KEY`
  - value: `<YOUR_GEMINI_API_KEY>`
- leave the rest of the configuration at its default settings

3. Once the VM instance is running, open an SSH session in the browser. Clone this repository, then navigate to the serving-rag-gemini folder.

```bash
git clone https://github.com/sruxll/rag_us_congressional_bills.git
cd rag_us_congressional_bills/serving-rag-gemini
```

4. Build the Docker Image

```bash
./docker-startup build
```

6. Run the App

```bash
./docker-startup deploy-gcp
```
## How to Run the Streamlit App on Cloud Run 🚀

Prior to starting cloud deployment, please make sure you have completed the see inital steps outlined in the GCP section above.

### Prerequisites

- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)
- A valid Hugging Face access token

### 1. Set Up the Google Cloud Environment

Install the Google Cloud CLI:
```bash
curl https://sdk.cloud.google.com | bash
```

Initialize the CLI:
```bash
exec -l $SHELL
gcloud init
```

Follow the prompts to sign in and set configurations.

### 2. Set Project Properties

Locate your project ID:

- Visit the [API Console](https://console.developers.google.com/)
- Navigate to **Manage all projects** and copy your project ID

Set the project in your terminal:
```bash
gcloud config set project <YOUR_PROJECT_ID>
```

Enable the necessary APIs:
```bash
gcloud services enable run cloudbuild containerregistry
```

Authenticate your terminal:
```bash
gcloud auth login
```

### 3. Build the Docker Container

From the root of your project directory (where the Dockerfile is located), run:
```bash
gcloud builds submit --tag gcr.io/<YOUR_PROJECT_ID>/streamlit-app
```

### 4. Create and Configure the Gemini API Secret

Create the secret in Secret Manager:
```bash
echo -n "your-gemini-api-key-goes-here" | \
gcloud secrets create gemini-api-key \
  --data-file=- \
  --replication-policy=automatic
```

Grant Cloud Run access to the secret key:
```bash
gcloud projects add-iam-policy-binding <YOUR_PROJECT_ID> \
  --member="serviceAccount:$(gcloud projects describe <YOUR_PROJECT_ID> --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

To replace an invalid key use:
```bash
echo -n "your-new-api-key-here" | \
gcloud secrets versions add gemini-api-key --data-file=-
```

### 5. Deploy to Cloud Run

To deploy an existing container:
```bash
bash deploy.sh
```

To build and deploy in one step:
```bash
bash build_and_deploy.sh
```
#### To Deploy in terminal:
```bash
gcloud run deploy serving-rag-gemini \
  --image gcr.io/<YOUR_PROJECT_ID>/serving-rag-gemini \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --update-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --port 8080 \
  --memory=4Gi \
  --min-instances=1 \
  --timeout=600
```

## Monitoring Usage and Staying Within Credit Limits

To avoid unexpected charges:

### 1. Enable Billing Alerts

Create a budget and receive alerts when you approach a spending threshold.

- Visit the [Billing Budgets & Alerts page](https://console.cloud.google.com/billing)
- Select your billing account
- Click **Create Budget**
- Set a monthly threshold (e.g., $5, $10)
Set up email alerts to notify you when usage hits your desired limit. For stricter control, you can also receive notifications when you're a specific percentage away from your budget threshold.
  
