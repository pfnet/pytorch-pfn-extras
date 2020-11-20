#!/bin/bash -uex

nvidia-docker run --volume ${PWD}:/work --workdir /work pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime .flexci/test.sh
