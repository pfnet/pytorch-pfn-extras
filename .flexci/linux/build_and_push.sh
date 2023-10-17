#!/bin/bash -uex

TARGET="${1}"

docker_build_and_push() {
    IMAGE_NAME="${PPE_FLEXCI_IMAGE_NAME}:${TARGET}"

    if [[ "${PPE_FLEXCI_IMAGE_REBUILD}" == "1" ]]; then
        echo "Force rebuilding docker image."
        CACHE_IMAGE_NAME=""
    else
        CACHE_IMAGE_NAME="${IMAGE_NAME}"
    fi

    DOCKER_BUILDKIT=1 docker build \
        -t "${IMAGE_NAME}" \
        -f "$(dirname ${0})/Dockerfile" \
        --cache-from "${CACHE_IMAGE_NAME}" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        "$@" .

    if [ "${PPE_FLEXCI_IMAGE_PUSH}" = "0" ]; then
      echo "Skipping docker push."
    else
      docker push "${IMAGE_NAME}"
    fi
}

case "${TARGET}" in
    torch110 )
        # PyTorch 1.10 + Python 3.8
        docker_build_and_push \
            --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04" \
            --build-arg python_version="3.8.15" \
            --build-arg pip_install_torch_args="torch==1.10.* torchvision==0.11.* -f https://download.pytorch.org/whl/cu113/torch_stable.html" \
        ;;

    torch111 )
        # PyTorch 1.11 + Python 3.9
        docker_build_and_push \
            --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04" \
            --build-arg python_version="3.9.7" \
            --build-arg pip_install_torch_args="torch==1.11.* torchvision==0.12.* -f https://download.pytorch.org/whl/cu113/torch_stable.html" \
        ;;

    torch112 )
        # PyTorch 1.12 + Python 3.10
        docker_build_and_push \
            --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04" \
            --build-arg python_version="3.10.5" \
            --build-arg pip_install_torch_args="torch==1.12.* torchvision==0.13.* -f https://download.pytorch.org/whl/cu113/torch_stable.html" \
        ;;

    torch113 )
        # PyTorch 1.13 + Python 3.10
        docker_build_and_push \
            --build-arg base_image="nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04" \
            --build-arg python_version="3.10.5" \
            --build-arg pip_install_torch_args="torch==1.13.* torchvision==0.14.* -f https://download.pytorch.org/whl/cu117/torch_stable.html" \
        ;;

    torch200 )
        # PyTorch 2.0 + Python 3.10
        docker_build_and_push \
            --build-arg base_image="nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04" \
            --build-arg python_version="3.10.5" \
            --build-arg pip_install_torch_args="torch==2.0.* torchvision==0.15.* -f https://download.pytorch.org/whl/cu117/torch_stable.html" \
        ;;

    * )
        echo "${1}: Unknown test name."
        exit 1
        ;;
esac
