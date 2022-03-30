import os
import re
import sys
import time
import argparse


def format_device_instance_group(device):
    if device == 'cpu':
        return "instance_group [{count: 1 kind: KIND_CPU }]"
    if device.startswith('cuda'):
        _, device_ids = device.split(':')
        return "instance_group [{count: 1 kind: KIND_GPU gpus: [ %s ]}]" % device_ids


def format_xrt_configuration(xrt_type):
    if xrt_type is not None:
        return 'parameters { key: "xrt" value: {string_value: "%s"}}' % xrt_type
    return None


def format_output_filename(output_dir, model_name, device_type, xrt_type):
    output_filename = os.path.join(output_dir, "{}_{}_{}_speed.txt")
    return output_filename.format(model_name, device_type, xrt_type)


def format_report_filename(output_dir, device, xrt_type, report_type):
    report_filename = os.path.join(output_dir, "speed_test_{}_{}_{}.txt")
    return report_filename.format(device, str(xrt_type), report_type)


def format_model_dir(configuration, model_name):
    return os.path.join(configuration['repo_dir'], model_name + "_repo")


def format_model_config_filename(configuration, model_name):
    return os.path.join(format_model_dir(configuration, model_name), model_name, "config.pbtxt")


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", required=True, help="models to test")
    parser.add_argument("--device", default="cpu", help="speed test device, --device cuda:n0,n1,n2|cpu")
    parser.add_argument("--xrt", default=None, help="xrt, --xrt tensorrt|openvino")
    parser.add_argument('--env-file', default='', help='environment file')
    parser.add_argument('--http-port', default='8000', help='http port')
    arguments = parser.parse_args()
    arguments = vars(arguments)
    return arguments


def user_configuration(arguments):
    configuration = {}
    configuration.update(arguments)
    configuration['model_names'] = configuration['model_names'].split()
    configuration['tritonserver_boot_retry_times'] = 20
    configuration['perf_analyzer_retry_times'] = 20
    configuration['working_dir'] = os.getcwd()
    configuration['output_dir'] = 'speed_test_output'
    configuration['repo_dir'] = os.path.join(configuration['working_dir'], 'repos')
    configuration['serving_image'] = 'registry.cn-beijing.aliyuncs.com/oneflow/oneflow-serving:nightly'
    return configuration


def prepare_model_configuration(model_name, config_file, device_configuration, xrt_configuration):
    with open(config_file, 'w+') as f:
        f.write('name: "{}"'.format(model_name))
        f.write('\n')
        f.write('backend: "oneflow"')
        f.write('\n')
        f.write(device_configuration)
        f.write('\n')
        if xrt_configuration is not None:
            f.write(xrt_configuration)


def run_shell_command(command):
    print(command)
    return os.system(command)


def launch_tritonserver(configuration, model_repo_dir):
    docker_container_name = 'triton-server'
    docker_rm_container = 'docker container rm -f ' + docker_container_name

    extra_docker_args = ''
    extra_docker_args += ' --rm'
    extra_docker_args += ' --runtime=nvidia'
    extra_docker_args += ' --network=host'
    extra_docker_args += ' --name ' + docker_container_name
    extra_docker_args += ' -v{}:/models'.format(model_repo_dir)
    if configuration['env_file'] is not None and configuration['env_file'] != '':
        extra_docker_args += ' --env-file {}'.format(configuration['env_file'])

    lanuch_command = '/opt/tritonserver/bin/tritonserver'
    lanuch_command += ' --model-store /models'
    lanuch_command += ' --strict-model-config false'
    lanuch_command += ' --http-port ' + configuration['http_port']
 
    run_shell_command(docker_rm_container)
    run_shell_command("docker run {} {} {} > server.log 2>&1 &".format(extra_docker_args, configuration['serving_image'], lanuch_command))

    retry_time = 0
    command = "cat server.log | grep HTTPService"
    ret = run_shell_command(command)
    while ret != 0 and retry_time < configuration['tritonserver_boot_retry_times']:
        retry_time += 1
        time.sleep(1)
        ret = run_shell_command(command)
    if ret != 0:
        return False
    return True


def launch_perf_analyzer(configuration, model_name, output_filename):
    docker_container_name = 'triton-server-sdk'
    docker_rm_container = 'docker container rm -f ' + docker_container_name

    retry_time = 0
    window = 5000 + 10000 * retry_time

    extra_docker_args = ''
    extra_docker_args += ' --rm'
    extra_docker_args += ' --runtime=nvidia'
    extra_docker_args += ' --shm-size=2g'
    extra_docker_args += ' --network=host'
    extra_docker_args += ' --name ' + docker_container_name

    docker_image = "nvcr.io/nvidia/tritonserver:21.10-py3-sdk"

    lanuch_command = 'perf_analyzer'
    lanuch_command += ' -m ' + model_name
    lanuch_command += ' -p ' + str(window)
    lanuch_command += ' -u localhost:' + configuration['http_port']
    lanuch_command += ' --concurrency-range 1:4'
    lanuch_command += ' --percentile=95'

    command = "docker run {} {} {} > {}".format(extra_docker_args, docker_image, lanuch_command, output_filename)

    ret = run_shell_command(docker_rm_container)
    ret = run_shell_command(command)
    while ret != 0 and retry_time < configuration['perf_analyzer_retry_times']:
        retry_time += 1
        ret = run_shell_command(command)
    if ret != 0:
        return False
    return True


def speed_test_clean(model_config_filename):
    run_shell_command("docker container rm -f triton-server")
    run_shell_command("docker container rm -f triton-server-sdk")
    os.remove('server.log')
    os.remove(model_config_filename)


def generate_detailed_report(configuration, model_name, device, xrt_type):
    output_filename = format_output_filename(configuration['output_dir'], model_name, device, xrt_type)
    detail_report_filename = format_report_filename(configuration['output_dir'], device, xrt_type, 'detail')
    run_shell_command("echo {}_{} >> {}".format(model_name, str(xrt_type), detail_report_filename))
    run_shell_command("cat {} | tail -n 5 >> {}".format(output_filename, detail_report_filename))


def summary_speed_test_output(model_name, output_filename, summary_report_filename):
    whole_text = ""
    with open(output_filename, "r") as f:
        whole_text = f.readlines()
    if whole_text == "" or len(whole_text) == 0:
        with open(summary_report_filename, "a+") as f:
            f.write("| ")
            f.write(model_name)
            f.write(" | x |\n")
    last_line = whole_text[-1]
    pattern = "Concurrency: 4, throughput: (.*) infer/sec, latency (.*) usec"
    match_objs = re.match(pattern, last_line)
    if match_objs is None or len(match_objs.groups()) != 2:
        with open(summary_report_filename, "a+") as f:
            f.write("| ")
            f.write(model_name)
            f.write(" | x |\n")
    else:
        with open(summary_report_filename, "a+") as f:
            f.write("| ")
            f.write(model_name)
            f.write(" | ")
            f.write(match_objs.groups()[0])
            f.write(" |\n")


def generate_summary_report(configuration, model_name, device, xrt_type):
    output_filename = format_output_filename(configuration['output_dir'], model_name, device, xrt_type)
    summary_report_filename = format_report_filename(configuration['output_dir'], device, xrt_type, 'summary')
    summary_speed_test_output(model_name, output_filename, summary_report_filename)


if __name__ == "__main__":
    arguments = parse_command_line_arguments()
    configuration = user_configuration(arguments)
    os.makedirs(configuration['output_dir'], exist_ok=True)

    for model_name in configuration['model_names']:
        model_repo_dir = format_model_dir(configuration, model_name)
        model_config_filename = format_model_config_filename(configuration, model_name)
        device_configuration = format_device_instance_group(configuration['device'])
        xrt_configuration = format_xrt_configuration(configuration['xrt'])
        output_filename = format_output_filename(configuration['output_dir'], model_name, configuration['device'], configuration['xrt'])

        prepare_model_configuration(model_name, model_config_filename, device_configuration, xrt_configuration)
        if not launch_tritonserver(configuration, model_repo_dir):
            sys.exit('tritonserver launch failed')
        if not launch_perf_analyzer(configuration, model_name, output_filename):
            sys.exit('perf_analyzer launch failed')
        generate_detailed_report(configuration, model_name, configuration['device'], configuration['xrt'])
        generate_summary_report(configuration, model_name, configuration['device'], configuration['xrt'])
        speed_test_clean(model_config_filename)

