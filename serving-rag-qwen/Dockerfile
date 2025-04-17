FROM python:3.11-slim

WORKDIR /root

COPY requirements.txt ./

RUN apt update 
RUN apt-get install -y python3-pip vim git
RUN pip install -r requirements.txt

EXPOSE 8501
ENTRYPOINT ["./entrypoint.sh"]