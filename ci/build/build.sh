#!/usr/bin/env bash
set -euxo pipefail

# build oneflow
echo "${ONEFLOW_CI_SRC_DIR}"
echo "${ONEFLOW_CI_BUILD_DIR}"
echo "${ONEFLOW_CI_PYTHON_EXE}"
echo "${ONEFLOW_CI_CMAKE_INIT_CACHE}"
echo "${HTTP_PROXY}"
gcc --version
ld --version

# clean python dir
cd ${ONEFLOW_CI_SRC_DIR}
${ONEFLOW_CI_PYTHON_EXE} -m pip install -i https://mirrors.aliyun.com/pypi/simple --user -r ci/fixed-dev-requirements.txt
cd python
git clean -nXd -e \!dist -e \!dist/**
git clean -fXd -e \!dist -e \!dist/**

# cmake config
mkdir -p ${ONEFLOW_CI_BUILD_DIR}
cd ${ONEFLOW_CI_BUILD_DIR}
find ${ONEFLOW_CI_BUILD_DIR} -name CMakeCache.txt
find ${ONEFLOW_CI_BUILD_DIR} -name CMakeCache.txt -delete
if [ ! -f "$ONEFLOW_CI_CMAKE_INIT_CACHE" ]; then
    echo "$ONEFLOW_CI_CMAKE_INIT_CACHE does not exist."
    exit 1
fi
cmake -S ${ONEFLOW_CI_SRC_DIR} -C ${ONEFLOW_CI_CMAKE_INIT_CACHE} -DPython3_EXECUTABLE=${ONEFLOW_CI_PYTHON_EXE}
# cmake build
cd ${ONEFLOW_CI_BUILD_DIR}
cmake --build . -j 8

export ONEFLOW_BUILD=$(pwd)
export PYTHONPATH=$(pwd)/../python

# build oneflow-backend
git config --global http.proxy ${HTTP_PROXY}
git config --global https.proxy ${HTTP_PROXY}
cd ../../oneflow-backend
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install  -DTRITON_BACKEND_REPO_TAG=r$TRITON_VERSION -DTRITON_CORE_REPO_TAG=r$TRITON_VERSION -DTRITON_COMMON_REPO_TAG=r$TRITON_VERSION -G Ninja -DCMAKE_PREFIX_PATH=$ONEFLOW_BUILD/liboneflow_cpp/share -DTRITON_ENABLE_GPU=ON ..
ninja -j16

# install flowvision, run export model
cd ../../ci
pip3 install -r build/requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ./test/test_resnet50_oneflow/resnet50_oneflow
python3 export_model.py
