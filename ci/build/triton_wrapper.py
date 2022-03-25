#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from abc import ABCMeta, abstractmethod


class Processor(metaclass=ABCMeta):

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def clean(self):
        pass


class XrtProcessor(Processor):

    def __init__(self, model_name, model_path, xrt_type):
        super().__init__()
        self._model_name = model_name
        self._model_path = model_path
        self._xrt_type = xrt_type

    def process(self):
        config_file_path = os.path.join(self._model_path, 'config.pbtxt')
        if self._xrt_type == 'openvino':
            self._append_instance_group(config_file_path, 'cpu')
        elif self._xrt_type == 'tensorrt':
            self._append_instance_group(config_file_path, 'gpu')
        self._append_xrt_config(config_file_path, self._xrt_type)
        return super().process()
    
    def clean(self):
        return super().clean()

    def _append_instance_group(self, config_file_path, instance_group):
        with open(config_file_path, 'a+') as f:
            f.write('\n')
            if instance_group == 'gpu':
                f.write('instance_group [{count: 1 kind: KIND_GPU}]')
            elif instance_group == 'cpu':
                f.write('instance_group [{count: 1 kind: KIND_CPU}]')
            f.write('\n')

    def _append_xrt_config(self, config_file_path, xrt_type):
        with open(config_file_path, 'a+') as f:
            f.write('\n')
            f.write('parameters {key: "xrt" value: {string_value: "%s"}}' % xrt_type)
            f.write('\n')


class EmptyConfigProcessor(Processor):

    def __init__(self, model_repo):
        super().__init__()
        self._model_repo = model_repo
        self._empty_models_path = {}

        models = os.listdir(self._model_repo)
        for model in models:
            config_file = os.path.join(self._model_repo, model, 'config.pbtxt')
            if not os.path.exists(config_file):
                self._empty_models_path[model] = config_file

    def process(self):
        # generate default config
        for model in self._empty_models_path:
            with open(self._empty_models_path[model], 'w+') as f:
                f.write('name: "{}"\n'.format(model))
                f.write('backend: "oneflow"\n')

        return super().process()

    def clean(self):
        for model in self._empty_models_path:
            os.remove(self._empty_models_path[model])
        return super().clean()


class TritonWrapper(object):

    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser(description='triton_wrapper: a command line tool to help you configure your model')
        self._parser.add_argument('--enable-openvino', help='specify the model name that wants to enable openvino', action='append')
        self._parser.add_argument('--enable-tensorrt', help='specify the model name that wants to enable tensorrt', action='append')
        
        self._args = None
        self._unknown = None

        self._launch_command = None
        self._model_repos = []
        self._processors = []
        self._model_to_path = {}

    def prepare(self):
        self._parse()
        
        self._unknown.extend(['--strict-model-config', 'false'])
        self._launch_command = 'tritonserver ' + ' '.join(self._unknown)

        for option, argument in zip(self._unknown, self._unknown[1:] + [' ']):
            if option == '--model-repository' or option == '--model-store':
                self._model_repos.append(argument)

        self._collect_models()
        self._prepare_processor()

    def start(self):
        # for each option, do process
        for processor in self._processors:
            processor.process()

        # launch tritonserver using the rest options
        try:
            subprocess.run(self._launch_command.split())
        except KeyboardInterrupt:
            # do nothing
            pass

        # for each option, do clean
        for processor in reversed(self._processors):
            processor.clean()
    
    def _parse(self):
        self._args, self._unknown = self._parser.parse_known_args()
        self._args = vars(self._args)

    def _collect_models(self):
        for model_repo in self._model_repos:
            models = os.listdir(model_repo)
            for model in models:
                if model in self._model_to_path:
                    sys.exit(model + ' is not unique across all model repositories')
                self._model_to_path[model] = os.path.join(model_repo, model)

    def _prepare_processor(self):
        # empty config file
        empty_config_models = []
        for model_repo in self._model_repos:
            processor = EmptyConfigProcessor(model_repo)
            empty_config_models = list(processor._empty_models_path.keys())
            self._processors.append(processor)

        # xrt config: openvino
        openvino_models = self._args['enable_openvino']
        if openvino_models is not None:
            for model_name in openvino_models:
                if model_name not in self._model_to_path:
                    print(model_name, ' will be ignored because it is not exist in the repository')
                    continue
                if model_name not in empty_config_models:
                    print(model_name, ' will be ignored because model configuration exists')
                    continue
                self._processors.append(XrtProcessor(model_name, self._model_to_path[model_name], 'openvino'))

        # xrt config: tensorrt
        tensorrt_models = self._args['enable_tensorrt']
        if tensorrt_models is not None:
            for model_name in tensorrt_models:
                if model_name not in self._model_to_path:
                    print(model_name, ' will be ignored because it is not exist in the repository')
                    continue
                if model_name not in empty_config_models:
                    print(model_name, ' will be ignored because model configuration exists')
                    continue
                self._processors.append(XrtProcessor(model_name, self._model_to_path[model_name], 'tensorrt'))


if __name__ == '__main__':
    wrapper = TritonWrapper()
    wrapper.prepare()
    wrapper.start()

