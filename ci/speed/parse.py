import os
import re

SPEED_TEST_DETAILED = "speed_test_detailed.txt"
SPEED_TEST_SUMMARY = "speed_test_summary.txt"

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
    model_names = ["alexnet", "efficientnet_b7", "mobilenet_v3_large", "resnet50", 
                   "resnet101", "vgg19", "vit_base_patch16_224", "mlp_mixer_b16_224"]
    parse_speed(model_names)
