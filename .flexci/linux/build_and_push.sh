#!/bin/bash -uex

TARGET="${1}"

TEST_PIP_PACKAGES="
matplotlib tensorboard ipython ipywidgets pandas optuna onnxruntime
pytest flake8 pysen[lint] pytest-cov slack_sdk
"

docker_build_and_push() {
    IMAGE_NAME="${PPE_FLEXCI_IMAGE_NAME}:${TARGET}"

    if [[ "${PPE_FLEXCI_IMAGE_REBUILD}" == "1" ]]; then
        echo "Force rebuilding docker image."
        CACHE_IMAGE_NAME=""
    else
        CACHE_IMAGE_NAME="${IMAGE_NAME}"
    fi

    pushd "$(dirname ${0})"
    DOCKER_BUILDKIT=1 docker build \
        -t "${IMAGE_NAME}" \
        --cache-from "${CACHE_IMAGE_NAME}" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        "$@" .
    popd

    if [ "${PPE_FLEXCI_IMAGE_PUSH}" = "0" ]; then
      echo "Skipping docker push."
    else
      docker push "${IMAGE_NAME}"
    fi
}

case "${TARGET}" in
    torch19 )
        # PyTorch 1.9 + Python 3.9
        docker_build_and_push \
            --build-arg base_image="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04" \
            --build-arg python_version="3.9.5" \
            --build-arg pip_install_torch_args="torch==1.9.* torchvision==0.10.* -f https://download.pytorch.org/whl/cu102/torch_stable.html" \
            --build-arg pip_install_dep_args="cupy-cuda102 pytorch-ignite onnx ${TEST_PIP_PACKAGES}"
        ;;

    torch110 )
        # PyTorch 1.10 + Python 3.9
        docker_build_and_push \
            --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04" \
            --build-arg python_version="3.9.7" \
            --build-arg pip_install_torch_args="torch==1.10.* torchvision==0.11.* -f https://download.pytorch.org/whl/cu113/torch_stable.html" \
            --build-arg pip_install_dep_args="cupy-cuda113 pytorch-ignite onnx ${TEST_PIP_PACKAGES}"
        ;;

    torch111 )

        # PyTorch 1.11 + Python 3.9
        docker_build_and_push \
            --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04" \
            --build-arg python_version="3.9.7" \
            --build-arg pip_install_torch_args="torch==1.11.* torchvision==0.12.* -f https://download.pytorch.org/whl/cu113/torch_stable.html" \
            --build-arg pip_install_dep_args="cupy-cuda113 pytorch-ignite onnx ${TEST_PIP_PACKAGES}"
        ;;

    torch112 )
        # PyTorch 1.12 + Python 3.10
        docker_build_and_push \
            --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04" \
            --build-arg python_version="3.10.5" \
            --build-arg pip_install_torch_args="torch==1.12.* torchvision==0.13.* -f https://download.pytorch.org/whl/cu113/torch_stable.html" \
            --build-arg pip_install_dep_args="cupy-cuda113 pytorch-ignite onnx ${TEST_PIP_PACKAGES}"
        ;;

    * )
        echo "${1}: Unknown test name."
        exit 1
        ;;
esac
