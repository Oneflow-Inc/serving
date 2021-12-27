DIRS=(test_*/)

passed=0
failed=0
for dir in "${DIRS[@]}"; do
    echo -e "Running $dir...\n"
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
# return (( $failed == 0 ))
