#!/bin/bash

echo "Starting download dataset"
huggingface-cli download --repo-type dataset --local-dir . "joynae5/CongressionalBillsDS" 

# Start the streamlit server, blocking exit
echo "Starting the Streamlit server"
streamlit run app.py --server.port=80 --server.address=0.0.0.0 --logger.level=error