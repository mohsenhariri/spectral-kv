#!/bin/bash

file="./requirements.txt"

while IFS='=' read -r package versino
do
    pip install --upgrade "$package"
done <"$file"

pip freeze > requirements.txt

