#!/usr/bin/env bash
set -euxo pipefail

DIRS=($(pwd)/test/test_*/)

for dir in "${DIRS[@]}"; do
    echo -e "Running test: $dir...\n"
    (cd $dir && ./test.sh)
    rc=$?
    if (( $rc != 0 )); then
        echo -e "Failed: $dir\n"
        exit 1
    fi
done

exit 0
