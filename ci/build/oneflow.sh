#!/usr/bin/env bash
set -euxo pipefail

flag_file="$ONEFLOW_CI_BUILD_DIR/done"

if [ ! -f $flag_file ]; then
    /bin/bash ${ONEFLOW_CI_BUILD_SCRIPT}
    git -C $ONEFLOW_CI_SRC_DIR rev-parse HEAD > $flag_file 
else
    oneflow_head_built=$(<$flag_file)
    oneflow_head=$(git -C $ONEFLOW_CI_SRC_DIR rev-parse HEAD)
    if [ "$oneflow_head_built" != "$oneflow_head" ]; then
        /bin/bash ${ONEFLOW_CI_BUILD_SCRIPT}
        echo "$oneflow_head" > $flag_file
    else
        echo "Use build cache for oneflow."
    fi
fi
