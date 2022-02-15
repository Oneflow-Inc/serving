"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import argparse
from jinja2 import Environment, FileSystemLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batching", action="store_true", help="Whether to use batching"
    )
    parser.add_argument(
        "-t",
        "--template",
        required=True,
        type=str,
        help="Pbtxt file template",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Where to save the outputed pbtxt",
    )
    args = parser.parse_args()
    env = Environment(loader=FileSystemLoader(os.path.dirname(args.template)))
    template = env.get_template(os.path.basename(args.template))
    content = template.render(batching=args.batching)
    with open(f"{args.output}/config.pbtxt", "w") as fp:
        fp.write(content)
