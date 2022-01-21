#!/usr/bin/env bash
set -euxo pipefail

export CUDA_VISIBLE_DEVICES=0

rm -rf ./models
mkdir ./models
cp -r ./resnet50_oneflow ./models
cp -r ./resnet50_oneflow ./models/resnet50_oneflow_batching
(cd ./models/resnet50_oneflow_batching && \
        sed -i "s/^name:.*/name: \"resnet50_oneflow_batching\"/" config.pbtxt && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 5/" config.pbtxt && \
        echo "dynamic_batching { max_queue_delay_microseconds: 1000000 }" >> config.pbtxt)

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
(cd ./models/resnet50_oneflow/ && python3 resnet50_test.py --model resnet50_oneflow)

echo "running resnet50 batching test"
(cd ./models/resnet50_oneflow_batching/ && python3 resnet50_test.py --model resnet50_oneflow_batching)


kill $SERVER_PID
wait $SERVER_PID

exit 0
