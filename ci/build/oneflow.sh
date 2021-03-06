#!/usr/bin/env bash
set -euxo pipefail

flag_file="$ONEFLOW_CI_BUILD_DIR/done"
export_pythonpath_script="$ONEFLOW_CI_BUILD_DIR/ci_source.sh"
oneflow_head=$(git -C $ONEFLOW_CI_SRC_DIR rev-parse HEAD)
whl_src_dir=$ONEFLOW_CI_SRC_DIR/python/dist

build_oneflow() {
    rm -rf $WHEELHOUSE_DIR
    mkdir -p $WHEELHOUSE_DIR
    /bin/bash ${ONEFLOW_CI_BUILD_SCRIPT}
    whls=$(ls $whl_src_dir)
    whls_arr=(${whls// /})
    whls_len=${#whls_arr[*]}
    if [ $whls_len == 1 ]; then
        cp $whl_src_dir/$whls $WHEELHOUSE_DIR/
    else
        echo "Please clean $whl_src_dir first"
        exit 1
    fi
    echo "$oneflow_head" > $flag_file
    echo "export PYTHONPATH=$ONEFLOW_CI_SRC_DIR/python" > $export_pythonpath_script
}

if [ ! -f $flag_file ]; then
    build_oneflow
else
    oneflow_head_built=$(<$flag_file)
    if [ "$oneflow_head_built" != "$oneflow_head" ]; then
        build_oneflow
    else
        cached_whl=$(ls $WHEELHOUSE_DIR)
        python3 -m pip install $WHEELHOUSE_DIR/$cached_whl
        > $export_pythonpath_script
        echo "Use build cache for oneflow."
    fi
fi
