FROM jinaai/jina:2.0-py37-perf

RUN apt-get update && apt install -y git

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
