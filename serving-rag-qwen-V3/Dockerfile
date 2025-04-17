FROM continuumio/miniconda3

WORKDIR /root

COPY requirements.txt ./


RUN conda create -y -n rag python=3.10
RUN echo "conda activate rag" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

RUN source activate rag
RUN conda install -c pytorch -c nvidia faiss-gpu && pip install -r requirements.txt

EXPOSE 80
ENTRYPOINT ["./entrypoint.sh"]