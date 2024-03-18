FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get install -y python3-pip \
    cmake 
RUN python3 -m pip install scikit-build
RUN python3 -m pip install numpy
RUN apt-get install -y gstreamer-1.0 \
     gir1.2-gst-rtsp-server-1.0  \
     python3-gi \
     iputils-ping \
     python3-gst-1.0 \
     libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     cmake \
     pkg-config
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gir1.2-gst-rtsp-server-1.0

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gstreamer1.0-rtsp
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN apt-get install -y libgirepository1.0-dev \
    gobject-introspection gir1.2-gst-rtsp-server-1.0 \
    python3-numpy

RUN python3 -m pip install pyds_ext
RUN python3 -m pip install cupy==12.3.0



ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.9.0

RUN apt-get update && \
    # Install build tools, build dependencies and python
    apt-get install -y \
	python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}


RUN python3 -m pip install python-socketio
RUN python3 -m pip install requests


ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages/:$PYTHONPATH
WORKDIR /opt/nvidia/deepstream/deepstream-6.3/sources/src
RUN apt-get update
# Install system packages
RUN apt-get install -y \
  # Base (min for TensorFlow)
    build-essential \
    checkinstall \
    cmake \
    curl \
    g++ \
    gcc \
    git \
    perl \
    pkg-config \
    protobuf-compiler \
    python3-dev \
    rsync \
    unzip \
    wget \
    zip \
    zlib1g-dev \
  # OpenCV
    doxygen \
    file \
    gfortran \
    gnupg \
    gstreamer1.0-plugins-good \
    imagemagick \
    libatk-adaptor \
    libatlas-base-dev \
    libboost-all-dev \
    libcanberra-gtk-module \
    libdc1394-22-dev \
    libeigen3-dev \
    libfaac-dev \
    libfreetype6-dev \
    libgflags-dev \
    libglew-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgoogle-glog-dev \
    libgphoto2-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    liblapack-dev \
    libmp3lame-dev \
    libopenblas-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libopenjp2-7 \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libopenjpip-server \
    libpng-dev \
    libpostproc-dev \
    libprotobuf-dev \
    libtbb2 \
    libtbb-dev \
    libtheora-dev \
    libtiff5-dev \
    libv4l-dev \
    libvorbis-dev \
    libwebp-dev \
    libx264-dev \
    libx265-dev \
    libxi-dev \
    libxine2-dev \
    libxmu-dev \
    libxvidcore-dev \
    libzmq3-dev \
    v4l-utils \
    x11-apps \
    x264 \
    yasm \
  # Torch
    libomp-dev \
    libsox-dev \
    libsox-fmt-all \
    libsphinxbase-dev \
    sphinxbase-utils \
    zlib1g \
  # FFMpeg (source install, do not install packages: libavcodec-dev libavformat-dev libavresample-dev libavutil-dev libswscale-dev)
    libass-dev \
    libc6 \
    libc6-dev \
    libnuma1 \
    libnuma-dev \
    libopus-dev \
    libtool \
    libvpx-dev \
  && apt-get clean

# Additional specialized apt installs
ARG CTO_CUDA_APT
ARG CTO_CUDA11_APT_XTRA
RUN apt-get install -y --no-install-recommends ${CTO_CUDA11_APT_XTRA} \
      time ${CTO_CUDA_APT} \
  && apt-get clean

# CUPTI library needed by TensorFlow1 but sometimes not in default path, adding if at unconventional location
# Also set the /tmp/.{CPU,GPU}_build file
ARG CTO_BUILD
COPY tools/cupti_helper.sh /tmp/
RUN chmod +x /tmp/cupti_helper.sh \
  && /tmp/cupti_helper.sh ${CTO_BUILD} \
  && rm /tmp/cupti_helper.sh

# Prepare ldconfig
RUN mkdir -p /usr/local/lib \
  && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/usrlocallib.conf' \
  && ldconfig

# Setup pip
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip3 install -U pip \
  && rm /tmp/get-pip.py

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Install Python tools (for buiding) 
ARG CTO_TF_NUMPY=numpy
RUN pip3 install -U mock
RUN pip3 install -U ${CTO_TF_NUMPY}
RUN pip3 install -U setuptools
RUN pip3 install -U six
RUN pip3 install -U wheel
RUN pip3 install -U future
RUN pip3 install -U packaging
RUN pip3 install -U Pillow
RUN pip3 install -U lxml
# RUN pip3 install -U pyyaml
RUN pip3 install -U mkl
RUN pip3 install -U mkl-include
RUN pip3 install -U cmake
RUN pip3 install -U cffi
RUN pip3 install -U typing
RUN pip3 install -U ninja
RUN pip3 install -U scikit-image
RUN pip3 install -U scikit-learn
RUN pip3 install -U keras_applications --no-deps
RUN pip3 install -U keras_preprocessing --no-deps
RUN rm -rf /root/.cache/pip



##### TensorFlow

## Download & Building TensorFlow from source in same RUN
ARG LATEST_BAZELISK=1.5.0
ARG LATEST_BAZEL=3.7.2
ARG CTO_TENSORFLOW_VERSION=2.7.0
ARG PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ARG CTO_TF_OPT=""
ARG CTO_DNN_ARCH=""
ARG TF_NEED_CLANG=0
RUN touch /tmp/.GPU_build
COPY tools/tf_build.sh /tmp/
RUN curl -s -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v${LATEST_BAZELISK}/bazelisk-linux-amd64
RUN chmod +x /usr/local/bin/bazel
RUN mkdir -p /usr/local/src/tensorflow
WORKDIR /usr/local/src
RUN wget -q --no-check-certificate -c https://github.com/tensorflow/tensorflow/archive/v${CTO_TENSORFLOW_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/tensorflow
WORKDIR /usr/local/src/tensorflow
RUN fgrep _TF_MAX_BAZEL configure.py | grep '=' | perl -ne '$lb="'${LATEST_BAZEL}'";$brv=$1 if (m%\=\s+.([\d\.]+).$+%); sub numit{@g=split(m%\.%,$_[0]);return(1000000*$g[0]+1000*$g[1]+$g[2]);}; if (&numit($brv) > &numit($lb)) { print "$lb" } else {print "$brv"};' > .bazelversion
RUN bazel clean
RUN chmod +x /tmp/tf_build.sh
RUN time /tmp/tf_build.sh Av1
RUN time ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN time pip3 install /tmp/tensorflow_pkg/tensorflow-*.whl
# RUN rm -rf /usr/local/src/tensorflow /tmp/tensorflow_pkg /tmp/bazel_check.pl /tmp/tf_build.sh /tmp/hsperfdata_root /root/.cache/bazel /root/.cache/pip /root/.cache/bazelisk
# RUN python3 -c "import tensorflow"



