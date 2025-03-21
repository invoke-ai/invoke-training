FROM ubuntu:22.04
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
        python3 python3-pip python3-venv ca-certificates \
        git libgl1 libglib2.0-0 git-lfs
WORKDIR /root/
RUN git clone https://github.com/invoke-ai/invoke-training.git
WORKDIR /root/invoke-training/
RUN pip uninstall numpy --yes && python3 -m venv venv && . venv/bin/activate && \
    python -m pip install --upgrade pip && \
    pip install . --extra-index-url https://download.pytorch.org/whl/cu126 \
    && pip3 install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu126
ENV PATH="$PATH:/root/invoke-training/venv/bin"
CMD ["sleep", "infinity"]
