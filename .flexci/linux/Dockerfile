# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ARG base_image
FROM ${base_image}

# Update GPG repository key.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install pyenv requirements.
# https://github.com/pyenv/pyenv/wiki/Common-build-problems#requirements
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y install \
        build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev git cmake protobuf-compiler libprotobuf-dev \
        openmpi-bin openmpi-common libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install pyenv.
RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT=/opt/pyenv
ENV PATH ${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

# Install Python.
ARG python_version
RUN pyenv install ${python_version} && \
    pyenv global ${python_version}

COPY ./tests/requirements.txt ./tests/requirements.mpi.txt ./

# Install test dependencies.
ARG pip_install_torch_args
ARG pip_install_dep_args	
RUN pip install -U pip && \
    pip install -U setuptools && \
    pip install -r requirements.txt -r requirements.mpi.txt ${pip_install_torch_args} && \
    pip install ${pip_install_dep_args} && \	
    pip list
