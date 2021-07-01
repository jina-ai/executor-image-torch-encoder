FROM jinaai/jina:2.0 as base

COPY . ./image_torch_encoder/
WORKDIR ./image_torch_encoder

RUN pip install .

FROM base
RUN pip install -r tests/requirements.txt
RUN pytest -s -v tests

FROM base
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
