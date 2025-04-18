FROM python:3.11-slim

WORKDIR /root

COPY requirements.txt ./

RUN apt update 
RUN apt-get install -y python3-pip git
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install google-genai

EXPOSE 80
ENTRYPOINT ["./entrypoint.sh"]
