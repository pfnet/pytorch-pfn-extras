#!/bin/bash -uex

nvidia-docker run --volume ${PWD}:/work --workdir /work nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 .flexci/test.sh
