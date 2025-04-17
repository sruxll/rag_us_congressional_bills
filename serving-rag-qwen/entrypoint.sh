#!/bin/bash

huggingface-cli download --repo-type dataset --local-dir . "joynae5/CongressionalBillsDS" 
huggingface-cli download --repo-type model "Qwen/Qwen2.5-1.5B-Instruct"

# Start the streamlit server, blocking exit
echo "Starting the Streamlit server"
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
