FROM nvidia/cuda:10.1-devel-ubuntu18.04
LABEL maintainer="r.chisholm@sheffield.ac.uk"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Cmake apt-get
RUN apt-get update --fix-missing && apt-get install -y \
    apt-transport-https \
    ca-certificates gnupg \
    software-properties-common \
    wget


# CMake signing key
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

# Install some basic packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential \
    make \
    cmake \
    doxygen \
    graphviz \
    python3-pip \
    valgrind

RUN pip3 install cpplint

CMD ["/bin/bash"]
WORKDIR /stage