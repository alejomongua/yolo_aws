#!/bin/bash
# deploy_model.sh
# This script creates a SageMaker model, endpoint configuration, and deploys an endpoint.
# Adjust the container image and model data location as needed.

set -e

# ***** USER-SPECIFIC PARAMETERS *****
REGION="us-east-1"
export AWS_PROFILE=admin
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
# Replace this with your SageMaker execution role ARN that has the necessary permissions.
ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/SageMakerExecutionRole"
MODEL_NAME="yolo-model"
ENDPOINT_CONFIG_NAME="yolo-endpoint-config"
ENDPOINT_NAME="yolo-endpoint"
# Replace with your inference container image. For example, the public PyTorch image:
CONTAINER_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py38"
# Location of the model artifact from your training job:
MODEL_DATA="s3://sagemaker-20250212110251/models/yolo/model.tar.gz"
# **************************************

echo "Creating SageMaker model '$MODEL_NAME'..."
aws sagemaker create-model \
  --model-name $MODEL_NAME \
  --primary-container Image=$CONTAINER_IMAGE,ModelDataUrl=$MODEL_DATA \
  --execution-role-arn $ROLE_ARN \
  --region $REGION

echo "Creating endpoint configuration '$ENDPOINT_CONFIG_NAME'..."
aws sagemaker create-endpoint-config \
  --endpoint-config-name $ENDPOINT_CONFIG_NAME \
  --production-variants VariantName=AllTraffic,ModelName=$MODEL_NAME,InitialInstanceCount=1,InstanceType=ml.m5.xlarge \
  --region $REGION

echo "Creating endpoint '$ENDPOINT_NAME'..."
aws sagemaker create-endpoint \
  --endpoint-name $ENDPOINT_NAME \
  --endpoint-config-name $ENDPOINT_CONFIG_NAME \
  --region $REGION

echo "Endpoint creation started. To wait for it to be in service, run:"
echo "aws sagemaker wait endpoint-in-service --endpoint-name $ENDPOINT_NAME --region $REGION"
