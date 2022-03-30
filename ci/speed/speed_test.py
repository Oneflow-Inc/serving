import os
import re
import sys
import time
import argparse
import subprocess
import warnings


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


def format_match_string(match_str):
    if match_str is None or len(match_str) == 0:
        return 'x | x | x | x | '
    res = ''
    for one_str in match_str:
        res += (one_str[0] + '(' + one_str[1] + ') | ')
    return res


def speed_test_header(configuration, device, xrt_type):
    header = '| Model | Concurrency(1) | Concurrency(2) | Concurrency(3) | Concurrency(4) |\n'
    sperators = '| ---- | ----  | ---- | ---- | ---- |\n'
    summary_report_filename = format_report_filename(configuration['output_dir'], device, xrt_type, 'summary')
    with open(summary_report_filename, 'w+') as f:
        f.write(header)
        f.write(sperators)


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
    configuration['tritonserver_boot_wait_time'] = 20
    configuration['perf_analyzer_wait_time'] = 120
    configuration['working_dir'] = os.getcwd()
    configuration['output_dir'] = 'speed_test_output'
    configuration['repo_dir'] = os.path.join(configuration['working_dir'], 'repos')
    configuration['tritonserver_image'] = 'registry.cn-beijing.aliyuncs.com/oneflow/oneflow-serving:nightly'
    configuration['tritonserver_sdk_image'] = '"nvcr.io/nvidia/tritonserver:21.10-py3-sdk"'
    configuration['tritonserver_container_name'] = 'triton-server'
    configuration['perf_analyzer_container_name'] = 'triton-server-sdk'
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


def run_shell_command(command, timeout=120):
    proc = subprocess.Popen(command, shell=True)
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        return 1
    return proc.returncode


def wait_tritonserver_launch(configuration):
    command = "cat server.log | grep HTTPService"
    ret = run_shell_command(command)
    retry_time = 0
    while ret != 0 and retry_time < configuration['tritonserver_boot_wait_time']:
        retry_time += 1
        time.sleep(1)
        ret = run_shell_command(command)
    return ret


def launch_tritonserver(configuration, model_repo_dir):
    docker_rm_container = 'docker container rm -f ' + configuration['tritonserver_container_name']

    extra_docker_args = ''
    extra_docker_args += ' --rm'
    extra_docker_args += ' --runtime=nvidia'
    extra_docker_args += ' --network=host'
    extra_docker_args += ' --name ' + configuration['tritonserver_container_name']
    extra_docker_args += ' -v{}:/models'.format(model_repo_dir)
    if configuration['env_file'] is not None and configuration['env_file'] != '':
        extra_docker_args += ' --env-file {}'.format(configuration['env_file'])

    lanuch_command = '/opt/tritonserver/bin/tritonserver'
    lanuch_command += ' --log-verbose 1'
    lanuch_command += ' --model-store /models'
    lanuch_command += ' --strict-model-config false'
    lanuch_command += ' --http-port ' + configuration['http_port']
 
    run_shell_command(docker_rm_container)
    run_shell_command("docker run {} {} {} > server.log 2>&1 &".format(extra_docker_args, configuration['tritonserver_image'], lanuch_command))

    if wait_tritonserver_launch(configuration) != 0:
        return False
    return True


def launch_perf_analyzer(configuration, model_name, output_filename):
    docker_rm_container = 'docker container rm -f ' + configuration['perf_analyzer_container_name']

    extra_docker_args = ''
    extra_docker_args += ' --rm'
    extra_docker_args += ' --runtime=nvidia'
    extra_docker_args += ' --shm-size=2g'
    extra_docker_args += ' --network=host'
    extra_docker_args += ' --name ' + configuration['perf_analyzer_container_name']

    lanuch_command = 'perf_analyzer'
    lanuch_command += ' -m ' + model_name
    lanuch_command += ' -u localhost:' + configuration['http_port']
    lanuch_command += ' --concurrency-range 1:4'
    lanuch_command += ' --percentile=95'

    command = "docker run {} {} {} > {}".format(extra_docker_args, configuration['tritonserver_sdk_image'], lanuch_command, output_filename)

    ret = run_shell_command(docker_rm_container)
    ret = run_shell_command(command, configuration['perf_analyzer_wait_time'])
    if ret != 0:
        return False
    return True


def speed_test_clean(model_config_filename):
    run_shell_command("docker container rm -f triton-server")
    run_shell_command("docker container rm -f triton-server-sdk")
    if os.path.exists('server.log'):
        os.remove('server.log')
    if os.path.exists(model_config_filename):
        os.remove(model_config_filename)


def summary_speed_test_output(model_name, output_filename, summary_report_filename):
    whole_text = ''
    with open(output_filename, 'r') as f:
        whole_text = f.readlines()
        whole_text = ''.join(whole_text)
    pattern = re.compile(r'Concurrency: .*, throughput: (.*) infer/sec, latency (.*) usec')
    match_objs = pattern.findall(whole_text)
    with open(summary_report_filename, 'a+') as f:
        f.write('| ')
        f.write(model_name)
        f.write(' | ')
        f.write(format_match_string(match_objs))
        f.write(' \n')


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
            warnings.warn('perf_analyzer launch failed')
        generate_summary_report(configuration, model_name, configuration['device'], configuration['xrt'])
        speed_test_clean(model_config_filename)

