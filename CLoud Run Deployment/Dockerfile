FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean

# Copy and install requirements
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Copy all source files
COPY . .

# Port 
ENV PORT=8080

# Run Streamlit 
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
