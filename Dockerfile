FROM jinaai/jina:master as base

COPY . ./image_torch_encoder/
WORKDIR ./image_torch_encoder

RUN pip install .

FROM base
RUN pip install -r tests/requirements.txt
RUN pytest tests

FROM base
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]