FROM continuumio/miniconda3

WORKDIR /root

COPY requirements.txt ./

RUN /opt/conda/bin/conda create -y -n rag python=3.10
RUN /opt/conda/bin/conda activate rag && /opt/conda/bin/conda install -c pytorch -c nvidia faiss-gpu && pip install -r requirements.txt


EXPOSE 80
ENTRYPOINT ["./entrypoint.sh"]
