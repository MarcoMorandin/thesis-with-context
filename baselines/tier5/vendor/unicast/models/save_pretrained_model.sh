#!/bin/bash

BASE_DIR=("./Timer/" "./Chronos/" "./CLIP/" "./BLIP/" "./Qwen/" "./LLaMA/")
TARGET_FILE="save_pretrained_model.py"

for dir in "${BASE_DIR[@]}" ; do
    if [ -d "$dir" ] && [ -f "$dir$TARGET_FILE" ]; then
        echo "$dir$TARGET_FILE"
        python "$dir$TARGET_FILE"
    fi
done