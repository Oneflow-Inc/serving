#!/usr/bin/env bash
set -euxo pipefail

export CUDA_VISIBLE_DEVICES=0

rm -rf ./models
mkdir -p models/resnet50/1
cp -r ../common/model models/resnet50/1/

# generate minimal config.pbtxt
echo "name: \"resnet50\"" >> models/resnet50/config.pbtxt
echo "backend: \"oneflow\"" >> models/resnet50/config.pbtxt

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1 --strict-model-config false"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

echo "running resnet50 basic test"
python3 ../common/test_model.py --model resnet50 --target-output ../common/resnet50_output.npy

kill $SERVER_PID
wait $SERVER_PID

exit 0
