#!/bin/bash -uex

nvidia-docker run --volume ${PWD}:/work --workdir /work pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime .flexci/test.sh
