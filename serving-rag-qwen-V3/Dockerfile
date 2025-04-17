FROM continuumio/miniconda3

WORKDIR /root

COPY requirements.txt ./

RUN conda create -y -n rag python=3.10
RUN conda activate rag && conda install -c pytorch -c nvidia faiss-gpu && pip install -r requirements.txt


EXPOSE 80
ENTRYPOINT ["./entrypoint.sh"]
