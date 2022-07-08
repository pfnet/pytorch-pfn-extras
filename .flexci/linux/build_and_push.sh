#!/bin/bash -uex

IMAGE_BASE="${1:-}"
IMAGE_PUSH=1
if [ "${IMAGE_BASE}" = "" ]; then
  IMAGE_BASE="pytorch-pfn-extras"
  IMAGE_PUSH=0
fi

TEST_PIP_PACKAGES="
matplotlib tensorboard ipython ipywidgets pandas optuna onnxruntime
pytest flake8 pysen[lint] pytest-cov
"

docker_build_and_push() {
    IMAGE_TAG="${1}"; shift
    IMAGE_NAME="${IMAGE_BASE}:${IMAGE_TAG}"

    pushd "$(dirname ${0})"
    docker build -t "${IMAGE_NAME}" "$@" .
    popd

    if [ "${IMAGE_PUSH}" = "0" ]; then
      echo "Skipping docker push."
    else
      docker push "${IMAGE_NAME}"
    fi
}

WAIT_PIDS=""

# PyTorch 1.8 + Python 3.6
docker_build_and_push torch18 \
    --build-arg base_image="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04" \
    --build-arg python_version="3.6.9" \
    --build-arg pip_install_torch_args="torch==1.8.* torchvision==0.9.* -f https://download.pytorch.org/whl/cu102/torch_stable.html" \
    --build-arg pip_install_dep_args="cupy-cuda102 pytorch-ignite onnx==1.11.0 ${TEST_PIP_PACKAGES}" \
    &
WAIT_PIDS="$! ${WAIT_PIDS}"

# PyTorch 1.9 + Python 3.9
docker_build_and_push torch19 \
    --build-arg base_image="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04" \
    --build-arg python_version="3.9.5" \
    --build-arg pip_install_torch_args="torch==1.9.* torchvision==0.10.* -f https://download.pytorch.org/whl/cu102/torch_stable.html" \
    --build-arg pip_install_dep_args="cupy-cuda102 pytorch-ignite onnx ${TEST_PIP_PACKAGES}" \
    &
WAIT_PIDS="$! ${WAIT_PIDS}"

# PyTorch 1.10 + Python 3.9
docker_build_and_push torch110 \
    --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04" \
    --build-arg python_version="3.9.7" \
    --build-arg pip_install_torch_args="torch==1.10.* torchvision==0.11.* -f https://download.pytorch.org/whl/cu113/torch_stable.html" \
    --build-arg pip_install_dep_args="cupy-cuda113 pytorch-ignite onnx ${TEST_PIP_PACKAGES}" \
    &
WAIT_PIDS="$! ${WAIT_PIDS}"

# Wait until the build complete.
for P in ${WAIT_PIDS}; do
    wait ${P}
done
WAIT_PIDS=""

# PyTorch 1.11 + Python 3.9
docker_build_and_push torch111 \
    --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04" \
    --build-arg python_version="3.9.7" \
    --build-arg pip_install_torch_args="torch==1.11.* torchvision==0.12.* -f https://download.pytorch.org/whl/cu113/torch_stable.html" \
    --build-arg pip_install_dep_args="cupy-cuda113 pytorch-ignite onnx ${TEST_PIP_PACKAGES}" \
    &
WAIT_PIDS="$! ${WAIT_PIDS}"

# PyTorch 1.12 + Python 3.9
docker_build_and_push torch112 \
    --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04" \
    --build-arg python_version="3.9.7" \
    --build-arg pip_install_torch_args="torch==1.12.* torchvision==0.13.* -f https://download.pytorch.org/whl/cu113/torch_stable.html" \
    --build-arg pip_install_dep_args="cupy-cuda113 pytorch-ignite onnx ${TEST_PIP_PACKAGES}" \
    &
WAIT_PIDS="$! ${WAIT_PIDS}"

# Wait until the build complete.
for P in ${WAIT_PIDS}; do
    wait ${P}
done
