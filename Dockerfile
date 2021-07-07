FROM jinaai/jina:2.0.3

COPY . ./image_torch_encoder/
WORKDIR ./image_torch_encoder

RUN pip install . lz4

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
