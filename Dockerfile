FROM jinaai/jina:2.0.0

# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y gcc build-essential

# setup the workspace
COPY jinahub/image_encoder /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]