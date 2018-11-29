#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: <runner.sh> <host_ip> <port>"
    exit 1
fi

export FLASK_APP=app.py
flask run --host=$1 --port=$2