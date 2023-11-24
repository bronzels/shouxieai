#!/bin/bash

SOME_COMMAND=$1
echo "SOME_COMMAND:${SOME_COMMAND}"

# 最大重试次数
MAX_RETRY=5
if [ $# -gt 1 ]; then
MAX_RETRY=$2
fi
echo "MAX_RETRY:${MAX_RETRY}"

# 等待s数
WAIT_SEC=3
if [ $# -gt 2 ]; then
WAIT_SEC=$3
fi
echo "WAIT_SEC:${WAIT_SEC}"

# 当前重试次数
retry_count=0

# 循环执行命令，如果失败则重试
while true; do
    # 执行命令，可以将具体的命令替换为你需要执行的命令
    command_result=$(${SOME_COMMAND})

    # 检查命令的返回值，如果成功则退出循环
    if [ $? -eq 0 ]; then
        echo "Command succeeded!"
        break
    fi

    # 命令执行失败，检查是否已达到最大重试次数
    if [ $retry_count -ge $MAX_RETRY ]; then
        echo "Command failed after $MAX_RETRY attempts."
        exit 1
    fi

    # 命令执行失败，增加重试次数，等待一段时间后再次执行命令
    echo "Command failed, retrying in ${WAIT_SEC} seconds..."
    retry_count=$((retry_count+1))
    sleep ${WAIT_SEC}
done