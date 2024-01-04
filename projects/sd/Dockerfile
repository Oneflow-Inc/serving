FROM serving:final

RUN cd /opt && \
    /miniconda3/envs/py310/bin/python -m pip --no-cache-dir install "torch" \
    "transformers==4.27.1" \
    "diffusers[torch]==0.19.3" \
    "flowvision" \
    "huggingface_hub" && \
    git clone https://github.com/Oneflow-Inc/onediff.git && \
    cd onediff && \
    /miniconda3/envs/py310/bin/python -m pip install -e .
