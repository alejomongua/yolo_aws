#!/bin/bash
# destroy_resources.sh
# This script deletes the SageMaker notebook instance, its lifecycle configuration,
# and the security group that was created for this project.

set -e

# ***** USER-SPECIFIC PARAMETERS *****
PROJECT_NAME="yolo"
NOTEBOOK_INSTANCE_NAME="${PROJECT_NAME}-notebook"
LIFECYCLE_CONFIG_NAME="${PROJECT_NAME}-lifecycle-config"
SECURITY_GROUP_NAME="${PROJECT_NAME}-sg"
REGION="us-east-1"
# **************************************

# Get SageMaker Notebook Instance Status
smni_status=$(aws sagemaker describe-notebook-instance --notebook-instance-name $NOTEBOOK_INSTANCE_NAME --region $REGION --query "NotebookInstanceStatus" --output text)

# Check if the notebook instance in service
if [ "$smni_status" == "InService" ]; then
  echo "Stopping SageMaker Notebook Instance: $NOTEBOOK_INSTANCE_NAME..."
  aws sagemaker stop-notebook-instance \
    --notebook-instance-name $NOTEBOOK_INSTANCE_NAME \
    --region $REGION
  echo "Waiting for notebook instance to stop..."
  aws sagemaker wait notebook-instance-stopped \
    --notebook-instance-name $NOTEBOOK_INSTANCE_NAME \
    --region $REGION
  echo "Notebook instance stopped."
fi

echo "Deleting SageMaker Notebook Instance: $NOTEBOOK_INSTANCE_NAME..."
aws sagemaker delete-notebook-instance \
  --notebook-instance-name $NOTEBOOK_INSTANCE_NAME \
  --region $REGION

echo "Waiting for notebook instance deletion..."
aws sagemaker wait notebook-instance-deleted \
  --notebook-instance-name $NOTEBOOK_INSTANCE_NAME \
  --region $REGION
echo "Notebook instance deleted."

echo "Deleting lifecycle configuration: $LIFECYCLE_CONFIG_NAME..."
aws sagemaker delete-notebook-instance-lifecycle-config \
  --notebook-instance-lifecycle-config-name $LIFECYCLE_CONFIG_NAME \
  --region $REGION

export AWS_PROFILE=admin
echo "Deleting security group: $SECURITY_GROUP_NAME..."
# Retrieve the security group ID (assuming it was created in the default VPC)
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
         --query "Vpcs[0].VpcId" --output text --region $REGION)
SG_ID=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$SECURITY_GROUP_NAME Name=vpc-id,Values=$VPC_ID \
       --query "SecurityGroups[0].GroupId" --output text --region $REGION)
aws ec2 delete-security-group --group-id $SG_ID --region $REGION

echo "All resources have been cleaned up."
export AWS_PROFILE=sagemaker