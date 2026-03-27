#!/bin/bash

echo "Downloading dogs and cats sample..."
wget https://s3.amazonaws.com/fast-ai-sample/dogscats_sample.tgz

echo "Extracting..."
tar -xzf dogscats_sample.tgz
rm dogscats_sample.tgz

echo "Done! Dataset is in the 'dogscats_sample' folder (train and valid subfolders included)."