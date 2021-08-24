# This Dockerfile builds an image with all dependencies needed for Origami.
#
# This includes:  Ubuntu 20.04, CUDA 10.1, CUDNN7, and Tensorflow-GPU 2.1.2
#
# Parts of this Dockerfile were modified from the official NVIDIA CUDA Dockerfile repo:
#       https://gitlab.com/nvidia/container-images/cuda
#  Specifically, everything up through the installation of CUDNN7.
#
# Notice mandated by the NVIDIA Deep Learning Container License:
#  “This software contains source code provided by NVIDIA Corporation.”
#
# Other parts of this Dockerfile were modified from official Tensorflow Dockerfiles.
#       https://github.com/tensorflow/tensorflow/tree/452c18fc5dfd64baf7ffdf6443b4aba8b0cc8b5e/tensorflow/tools/dockerfiles

# The following was modified from the Dockerfile for NVIDIA CUDA 18.04 - base:
#       https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.1/ubuntu18.04/base/Dockerfile

FROM ubuntu:20.04 as base

#FROM base as base-amd64

LABEL maintainer "Jack Rasiel <jrasiel@umd.edu>"

ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
ENV NV_CUDA_CUDART_VERSION 10.1.243-1

ENV NV_ML_REPO_ENABLED 1
ENV NV_ML_REPO_URL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/${NVARCH}

# The following was modified from the Dockerfile for NVIDIA CUDA 18.04 - runtime:
#      https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.1/ubuntu18.04/runtime/Dockerfile

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/${NVARCH}/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    if [ ! -z ${NV_ML_REPO_ENABLED} ]; then echo "deb ${NV_ML_REPO_URL} /" > /etc/apt/sources.list.d/nvidia-ml.list; fi && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-10-1=${NV_CUDA_CUDART_VERSION} \
    cuda-compat-10-1 \
    && ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV NV_CUDA_LIB_VERSION 10.1.243-1
ENV NV_NVTX_VERSION 10.1.243-1
ENV NV_LIBNPP_VERSION 10.1.243-1
ENV NV_LIBCUSPARSE_VERSION 10.1.243-1


ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas10

ENV NV_LIBCUBLAS_VERSION 10.2.1.243-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}


ENV NV_LIBNCCL_PACKAGE_NAME "libnccl2"
ENV NV_LIBNCCL_PACKAGE_VERSION 2.8.3-1
ENV NCCL_VERSION 2.8.3
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda10.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-10-1=${NV_CUDA_LIB_VERSION} \
    cuda-npp-10-1=${NV_LIBNPP_VERSION} \
    cuda-nvtx-10-1=${NV_NVTX_VERSION} \
    cuda-cusparse-10-1=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBNCCL_PACKAGE_NAME} ${NV_LIBCUBLAS_PACKAGE_NAME}

# The following was modified from the Dockerfile for NVIDIA CUDA 18.04 - runtime - cudnn7:
#      https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.1/ubuntu18.04/runtime/cudnn7/Dockerfile 

ENV NV_CUDNN_PACKAGE_VERSION 7.6.5.32-1
ENV NV_CUDNN_VERSION 7.6.5.32

ENV NV_CUDNN_PACKAGE_NAME libcudnn7
ENV NV_CUDNN_PACKAGE ${NV_CUDNN_PACKAGE_NAME}=${NV_CUDNN_PACKAGE_VERSION}+cuda10.1

ENV CUDNN_VERSION ${NV_CUDNN_VERSION}

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} && \
    rm -rf /var/lib/apt/lists/*

# The following was modified from the Dockerfile for Tensorflow-GPU (commit 452c18fc5d):
# https://github.com/tensorflow/tensorflow/blob/452c18fc5dfd64baf7ffdf6443b4aba8b0cc8b5e/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA=10.1
ARG CUDNN=7.6.5.32-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# No user prompts during apt install:
ARG DEBIAN_FRONTEND=noninteractive

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        # There appears to be a regression in libcublas10=10.2.2.89-1 which
        # prevents cublas from initializing in TF. See
        # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
        libcublas10=10.2.1.243-1 \ 
        cuda-nvrtc-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        wget

# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

##############################################################
####### END OF TENSORFLOW DOCKERFILE-DERIVED MATERIAL ########
##############################################################

# The following lines install dependencies specific to Origami:
# Install Python3.7 from deadsnake ppa:
#  (We need Python3.7 for compatibility reasons)
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python3.7 \
    python3-pip 

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install setuptools

RUN apt update && apt install -y --no-install-recommends \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        tesseract-ocr \
        libtesseract-dev \
        libleptonica-dev \
        pkg-config \
        libcgal-dev \
        libopenmpi-dev \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libffi-dev \
        python3.7-dev \
        libcairo2-dev \
        git

# Install scikit-geometry from source.  This takes a while.
RUN git clone https://github.com/scikit-geometry/scikit-geometry
RUN echo "NOTE!  Building scikit-geometry from source.  This will take a while!" && \
        cd scikit-geometry && python3.7 setup.py install

# Pip install from pip freeze with --no-deps flag
COPY pip_requirements.txt /tmp/pip_requirements.txt
RUN python3.7 -m pip install -r /tmp/pip_requirements.txt --no-deps

# uninstall and reinstall h5py<3.0.0
RUN python3.7 -m pip uninstall --yes h5py && python3.7 -m pip install 'h5py<3.0.0'

# Some TF tools expect a "python" binary
#RUN ln -s $(which python3) /usr/local/bin/python
RUN ln -s $(which python3.7) /usr/local/bin/python

# Get the specific tflow-gpu wheel we need for Origami:
RUN wget https://files.pythonhosted.org/packages/0d/eb/9e03ca9b0b1d91274d9cfc90bfca7d75ff90df8e28160626f7b016f05b69/tensorflow_gpu-2.1.2-cp37-cp37m-manylinux2010_x86_64.whl

# Install TFlow GPU:
RUN python3.7 -m pip install tensorflow_gpu-2.1.2-cp37-cp37m-manylinux2010_x86_64.whl 

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

