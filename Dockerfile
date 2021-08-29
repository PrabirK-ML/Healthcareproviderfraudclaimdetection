FROM ubuntu
COPY . /home/prabir/Work/docker/
EXPOSE 5000
WORKDIR /home/prabir/Work/docker/
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
CMD python3.8 web.py

