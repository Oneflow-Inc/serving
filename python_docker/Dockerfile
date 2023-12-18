FROM nvcr.io/nvidia/tritonserver:23.10-py3

# 安装 oneflow, flowvison，torch, onediff 和 其他以来库-
RUN python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu122 && \
    python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.19.3" "flowvision" "huggingface_hub" && \
    cd /root && git clone https://github.com/Oneflow-Inc/onediff.git && cd onediff && python3 -m pip install -e .

# 安装 oneflow python backend
COPY ./model.py /opt/tritonserver/backends/oneflow/