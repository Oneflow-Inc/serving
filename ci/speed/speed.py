import os
import re
import sys


BOOT_RETRY_TIMES=20
PERF_RETRY_TIMES=10
SPEED_TEST_DETAILED = "speed_test_detailed.txt"
SPEED_TEST_SUMMARY = "speed_test_summary.txt"


# speed test
def speed_test(model_names, model_repo_root):
    for model_name in model_names:
        model_repo_dir = os.path.join(model_repo_root, model_name + "_repo")
        print(model_name, model_repo_dir)
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

        ret = os.system("docker container rm -f triton-server-sdk")

        retry_time = 0
        window = 5000 + 1000 * retry_time
        command = "docker run --rm --runtime=nvidia --shm-size=2g --network=host --name triton-server-sdk nvcr.io/nvidia/tritonserver:21.10-py3-sdk perf_analyzer -m {} --shape INPUT_0:3,224,224 -p {} -u localhost:8003 --concurrency-range 1:4  --percentile=95 > {}_speed.txt".format(model_name, str(window), model_name)
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


def parse_speed(model_names):
    for model_name in model_names:
        ret = os.system("echo {} >> {}".format(model_name, SPEED_TEST_DETAILED))
        ret = os.system("cat {}_speed.txt | tail -n 5 >> {}".format(model_name, SPEED_TEST_DETAILED))
        
        model_speed_file = "{}_speed.txt".format(model_name)
        whole_text = ""
        with open(model_speed_file, "r") as f:
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
            f.write(" | ")
            f.write("x | \n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('usage: python3 models.py "working_dir" "model_names"')
        exit()
    working_dir = os.path.join(sys.argv[1], "repos")
    model_names = sys.argv[2].split()
    speed_test(model_names, working_dir)
    parse_speed(model_names)
