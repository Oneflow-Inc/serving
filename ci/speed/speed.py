import os
import re
import shutil
import argparse


def get_device_configuration(device : str):
    if device == "cpu":
        return "cpu", "instance_group [{count: 1 kind: KIND_CPU }]"
    if device.startswith("cuda"):
        device, device_id = device.split(":")
        return device, "instance_group [{count: 1 kind: KIND_GPU gpus: [ %s ]}]" % device_id


parser = argparse.ArgumentParser()
parser.add_argument("--model_names", required=True, help="models to test")
parser.add_argument("--device", default="cpu", help="speed test device, --device cuda:n|cpu")
parser.add_argument("--xrt", default=None, help="xrt, --xrt tensorrt|openvino")


FLAGS = parser.parse_args()
MODEL_NAMES = FLAGS.model_names.split()
DEVICE = FLAGS.device
XRT_TYPE = FLAGS.xrt
BOOT_RETRY_TIMES=20
PERF_RETRY_TIMES=10
WORKING_DIR = os.getcwd()
XRT_CONFIGURATION = 'parameters { key: "xrt" value: {string_value: "%s"}}' % XRT_TYPE
OUTPUT_FILE_DIR = "speed_test_output"
OUTPUT_FILE_NAME = os.path.join(OUTPUT_FILE_DIR, "{}_{}_{}_speed.txt")
SPEED_TEST_DETAILED = os.path.join(OUTPUT_FILE_DIR, "speed_test_{}_{}_detailed.txt".format(DEVICE, XRT_TYPE))
SPEED_TEST_SUMMARY = os.path.join(OUTPUT_FILE_DIR, "speed_test_{}_{}_summary.txt".format(DEVICE, XRT_TYPE))
DEVICE, DEVICE_CONFIGURATION = get_device_configuration(DEVICE)


# speed test
def speed_test(model_names, model_repo_root, device="cpu", xrt_type=None):
    for model_name in model_names:
        model_repo_dir = os.path.join(model_repo_root, model_name + "_repo")
        config_pb_txt = os.path.join(model_repo_root, model_name + "_repo", model_name, "config.pbtxt")
        output_file_name = OUTPUT_FILE_NAME.format(model_name, device, str(xrt_type))

        shutil.copyfile(config_pb_txt, config_pb_txt + ".bak")
        with open(config_pb_txt, "a+") as f:
            f.write(DEVICE_CONFIGURATION)
            f.write("\n")
            if xrt_type is not None:
                f.write(XRT_CONFIGURATION)

        ret = os.system("docker container rm -f triton-server")
        ret = os.system("docker run --rm --name triton-server -v{}:/models --runtime=nvidia -p8003:8000 -p8001:8001 -p8002:8002 registry.cn-beijing.aliyuncs.com/oneflow/oneflow-serving:nightly /opt/tritonserver/bin/tritonserver --model-store /models --strict-model-config false > server.log 2>&1 &".format(model_repo_dir))

        retry_time = 0
        command = "cat server.log | grep HTTPService"
        ret = os.system(command)
        while ret != 0 and retry_time < BOOT_RETRY_TIMES:
            retry_time += 1
            ret = os.system("sleep 1")
            ret = os.system(command)
        if ret != 0:
            print("triton server boot failed")
            return

        retry_time = 0
        window = 5000 + 10000 * retry_time
        ret = os.system("docker container rm -f triton-server-sdk")
        command = "docker run --rm --runtime=nvidia --shm-size=2g --network=host --name triton-server-sdk nvcr.io/nvidia/tritonserver:21.10-py3-sdk perf_analyzer -m {} --shape INPUT_0:3,224,224 -p {} -u localhost:8003 --concurrency-range 1:4  --percentile=95 > {}".format(model_name, str(window), output_file_name)
        ret = os.system(command)
        while ret != 0 and retry_time < PERF_RETRY_TIMES:
            retry_time += 1
            ret = os.system(command)
        if ret != 0:
            print("perf_analyzer failed")
            return

        ret = os.system("docker container rm -f triton-server")
        ret = os.system("docker container rm -f triton-server-sdk")
        ret = os.system("rm server.log")
        ret = os.system("mv {}.bak {}".format(config_pb_txt, config_pb_txt))


def parse_speed(model_names, device="cpu", xrt_type=None):
    for model_name in model_names:
        output_file_name = OUTPUT_FILE_NAME.format(model_name, device, str(xrt_type))
        ret = os.system("echo {}_{} >> {}".format(model_name, str(xrt_type), SPEED_TEST_DETAILED))
        ret = os.system("cat {} | tail -n 5 >> {}".format(output_file_name, SPEED_TEST_DETAILED))

        whole_text = ""
        with open(output_file_name, "r") as f:
            whole_text = f.readlines()
        if whole_text == "" or len(whole_text) <= 0:
            continue
        last_line = whole_text[-1]
        pattern = "Concurrency: 4, throughput: (.*) infer/sec, latency (.*) usec"
        match_objs = re.match(pattern, last_line)
        if match_objs is None:
            continue
        else:
            match_objs = match_objs.groups()
        if len(match_objs) != 2:
            continue
        with open(SPEED_TEST_SUMMARY, "a+") as f:
            f.write(model_name)
            f.write(" | ")
            f.write(match_objs[0])
            f.write(" |\n")


if __name__ == "__main__":
    os.makedirs(OUTPUT_FILE_DIR, exist_ok=True)
    speed_test(MODEL_NAMES, os.path.join(WORKING_DIR, "repos"), DEVICE, XRT_TYPE)
    parse_speed(MODEL_NAMES, DEVICE, XRT_TYPE)
