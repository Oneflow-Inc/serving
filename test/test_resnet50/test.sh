#!/usr/bin/env bash
set -euxo pipefail

export CUDA_VISIBLE_DEVICES=0

rm -rf ./models
mkdir -p models/resnet50/1
mkdir -p models/resnet50_batching/1

cp -r ./model models/resnet50/1/
cp -r ./model models/resnet50_batching/1/

python3 ../common/generate_pbtxt.py --template ./config.pbtxt.j2 --output models/resnet50
python3 ../common/generate_pbtxt.py --template ./config.pbtxt.j2 --output models/resnet50_batching --batching

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

echo "running resnet50 basic test"
python3 ../common/test_model.py --model resnet50 --target-output ./resnet50_output.npy

echo "running resnet50 batching test"
python3 ../common/test_model.py --model resnet50_batching --target-output ./resnet50_output.npy


kill $SERVER_PID
wait $SERVER_PID

exit 0
