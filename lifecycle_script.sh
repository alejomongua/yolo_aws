#!/bin/bash
set -e
DATA_DIR="/home/ec2-user/SageMaker/datasets"
mkdir -p "${DATA_DIR}"
VOC_TAR="VOCtrainval_11-May-2012.tar"
VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/${VOC_TAR}"
if aws s3 ls s3://sagemaker-20250212110251/datasets/${VOC_TAR}; then
    echo "Dataset already exists in S3."
    aws s3 cp s3://sagemaker-20250212110251/datasets/${VOC_TAR} "${DATA_DIR}/${VOC_TAR}"
else
    echo "Downloading Pascal VOC dataset..."
    wget "${VOC_URL}" -P "${DATA_DIR}"
    echo "Uploading dataset to S3..."
    aws s3 cp "${DATA_DIR}/${VOC_TAR}" s3://sagemaker-20250212110251/datasets/${VOC_TAR}
fi
# Extract the dataset
tar -xf "${DATA_DIR}/${VOC_TAR}" -C "${DATA_DIR}"
# Download the YOLO implementation notebook
aws s3 cp s3://sagemaker-20250212110251/notebooks/implementation.ipynb /home/ec2-user/SageMaker/
