#!/usr/bin/env bash
set -aexo pipefail
SCRIPT_DIR="$( cd -P "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" && pwd )"
cd "${SCRIPT_DIR}"
FOLDER_NAME =$(basename "${SCRIPT_DIR}")
cleanup() {
    rm -rf tmp
    docker rm -f ${FOLDER_NAME}
}
trap cleanup EXIT
mkdir -p tmp
pushd tmp
    rm -rf nga-baroque
    git clone https://huggingface.co/datasets/InvokeAI/nga-baroque
    pushd nga-baroque
        sudo apt install git-lfs --yes && git lfs install
        git lfs pull
    popd
popd
docker build -t ${FOLDER_NAME} .
docker run \
    --rm -itd \
    --name ${FOLDER_NAME} \
    -h ${FOLDER_NAME} \
    -p 1234:1234 \
    -p 6006:6006 \
    --ipc=host \
    --gpus all \
    -v ${SCRIPT_DIR}/tmp:/root/invoke-training/data \
    ${FOLDER_NAME}
docker exec ${FOLDER_NAME} bash -c ". venv/bin/activate; python3 -c 'import torch; print(torch.ones(1,device=\"cuda:0\"))'"
docker exec ${FOLDER_NAME} bash -c 'invoke-train-ui --host 0.0.0.0 --port 1234'
