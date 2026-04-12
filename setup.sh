#!/bin/bash

python --version
if [ $? -ne 0 ]; then
    echo "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

pip --version
if [ $? -ne 0 ]; then
    echo "pip is not installed. Please install pip for Python 3."
    exit 1
fi

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt