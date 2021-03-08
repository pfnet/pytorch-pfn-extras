#!/bin/bash -uex

IMAGE_BASE="${1:-}"
IMAGE_PUSH=1
if [ "${IMAGE_BASE}" = "" ]; then
  IMAGE_BASE="pytorch-pfn-extras"
  IMAGE_PUSH=0
fi

TEST_PIP_PACKAGES="
matplotlib tensorboard ipython ipywidgets pandas optuna onnx
pytest flake8
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

# PyTorch 1.5 + Python 3.5
docker_build_and_push torch15 \
    --build-arg base_image="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04" \
    --build-arg python_version="3.5.9" \
    --build-arg pip_packages="torch==1.5.* torchvision==0.6.* cupy-cuda102 pytorch-ignite<0.4.3 ${TEST_PIP_PACKAGES}" &
WAIT_PIDS="$! ${WAIT_PIDS}"

# PyTorch 1.6 + Python 3.6
docker_build_and_push torch16 \
    --build-arg base_image="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04" \
    --build-arg python_version="3.6.12" \
    --build-arg pip_packages="torch==1.6.* torchvision==0.7.* cupy-cuda102 pytorch-ignite ${TEST_PIP_PACKAGES}" &
WAIT_PIDS="$! ${WAIT_PIDS}"

# PyTorch 1.7 + Python 3.8
docker_build_and_push torch17 \
    --build-arg base_image="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04" \
    --build-arg python_version="3.8.6" \
    --build-arg pip_packages="torch==1.7.* torchvision==0.8.* cupy-cuda102 pytorch-ignite ${TEST_PIP_PACKAGES}" &
WAIT_PIDS="$! ${WAIT_PIDS}"

# PyTorch 1.8 + Python 3.9
docker_build_and_push torch18 \
    --build-arg base_image="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04" \
    --build-arg python_version="3.9.2" \
    --build-arg pip_packages="torch==1.8.* torchvision==0.9.* cupy-cuda102 pytorch-ignite ${TEST_PIP_PACKAGES}" &
WAIT_PIDS="$! ${WAIT_PIDS}"

# Wait until the build complete.
for P in ${WAIT_PIDS}; do
    wait ${P}
done
