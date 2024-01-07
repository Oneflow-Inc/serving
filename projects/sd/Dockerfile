FROM serving:final

RUN /miniconda3/envs/py310/bin/python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu122

RUN cd /opt && \
    /miniconda3/envs/py310/bin/python -m pip --no-cache-dir install "torch" \
    "transformers==4.27.1" \
    "diffusers[torch]==0.19.3" \
    "flowvision" \
    "huggingface_hub" && \
    git clone https://github.com/siliconflow/onediff.git && \
    cd onediff && \
    /miniconda3/envs/py310/bin/python -m pip install -e .

RUN sed -i "/export PYTHONPATH=\/opt\/oneflow\/python/d" ~/.bashrc

COPY --chown=1000:1000 ./projects/sd/src/model.py /opt/tritonserver/backends/oneflow_sd/
COPY --chown=1000:1000 ./projects/sd/oneflow_sd_serving_triton_entrypoint.sh /opt/nvidia/

ENTRYPOINT ["/opt/nvidia/oneflow_sd_serving_triton_entrypoint.sh"]