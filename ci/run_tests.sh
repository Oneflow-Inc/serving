# Copyright 2020 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xe

./install_dependency.sh

DIRS=(test_*/)

passed=0
failed=0
for dir in "${DIRS[@]}"; do
    echo -e "Running test: $dir...\n"
    (cd $dir && ./test.sh)
    rc=$?
    if (( $rc == 0 )); then
        (( passed++ ))
    else
        echo -e "Failed\n"
        (( failed++ ))
    fi
done

echo -e "\n***\n***\nPassed: ${passed}\nFailed: ${failed}\n***\n***\n"
exit $failed
