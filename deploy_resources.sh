#!/bin/bash
# deploy_resources.sh
# This script sets up the SageMaker notebook instance (using free-tier eligible ml.t2.medium)
# with a security group restricting access to your current public IP, and a lifecycle
# configuration to download and upload the Pascal VOC dataset.

set -e

# ***** USER-SPECIFIC PARAMETERS *****
PROJECT_NAME="yolo"
NOTEBOOK_INSTANCE_NAME="${PROJECT_NAME}-notebook"
LIFECYCLE_CONFIG_NAME="${PROJECT_NAME}-lifecycle-config"
# This lifecycle script downloads the Pascal VOC2007 training/validation tarball and uploads it to S3.
LIFECYCLE_SCRIPT=lifecycle_script.sh
SECURITY_GROUP_NAME="${PROJECT_NAME}-sg"
BUCKET="sagemaker-20250212110251"
REGION="us-east-1"
export AWS_PROFILE=admin
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
# Replace this with your SageMaker execution role ARN that has the necessary permissions.
ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/SageMakerExecutionRole"
# **************************************

echo "Determining your public IP address..."
MY_IP=$(curl -s http://checkip.amazonaws.com)
echo "Your public IP is: $MY_IP"

echo "Getting the default VPC ID..."
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
         --query "Vpcs[0].VpcId" --output text --region $REGION)
echo "Default VPC ID: $VPC_ID"

# Check if the security group already exists
SG_ID=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$SECURITY_GROUP_NAME \
            --query "SecurityGroups[0].GroupId" --output text --region $REGION)
if [ "$SG_ID" == "None" ]; then
  echo "Creating security group '$SECURITY_GROUP_NAME'..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name ${SECURITY_GROUP_NAME} \
    --description "SG for ${PROJECT_NAME} notebook restricted to $MY_IP" \
    --vpc-id $VPC_ID --region $REGION \
    --query 'GroupId' --output text)
    echo "Created security group with ID: $SG_ID"
    echo "Adding inbound rule to allow HTTPS (port 443) from $MY_IP/32..."
    aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr ${MY_IP}/32 \
    --region $REGION
else
    echo "Security group '$SECURITY_GROUP_NAME' already exists with ID: $SG_ID"
fi

echo "Selecting a subnet from the default VPC..."
# Here we assume the first subnet is public and has a route to an Internet Gateway.
SUBNET_ID=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID \
           --query "Subnets[0].SubnetId" --output text --region $REGION)
echo "Using Subnet ID: $SUBNET_ID"

export AWS_PROFILE=sagemaker

# Check if instance lifecycle configuration already exists
CONFIG_LIST=$(aws sagemaker list-notebook-instance-lifecycle-configs --query 'NotebookInstanceLifecycleConfigs[*].NotebookInstanceLifecycleConfigName' --output text --region $REGION)
CONFIG_EXISTS=$(echo $CONFIG_LIST | grep -o $LIFECYCLE_CONFIG_NAME || echo "None")
if [ "$CONFIG_EXISTS" == "None" ]; then
    echo "Creating notebook instance lifecycle configuration..."

    # Base64 encode the script (the CLI expects base64-encoded content)
    ENCODED_SCRIPT=$(cat "$LIFECYCLE_SCRIPT" | base64 | tr -d '\n')

    aws sagemaker create-notebook-instance-lifecycle-config \
    --notebook-instance-lifecycle-config-name $LIFECYCLE_CONFIG_NAME \
    --on-start Content="$ENCODED_SCRIPT" \
    --region $REGION

    echo "Created lifecycle configuration: $LIFECYCLE_CONFIG_NAME"
else
    echo "Lifecycle configuration '$LIFECYCLE_CONFIG_NAME' already exists."
fi

echo "Creating SageMaker notebook instance $NOTEBOOK_INSTANCE_NAME..."
aws sagemaker create-notebook-instance \
  --notebook-instance-name $NOTEBOOK_INSTANCE_NAME \
  --instance-type ml.t3.medium \
  --role-arn $ROLE_ARN \
  --subnet-id $SUBNET_ID \
  --security-group-ids $SG_ID \
  --direct-internet-access Enabled \
  --lifecycle-config-name $LIFECYCLE_CONFIG_NAME \
  --region $REGION

echo "Notebook instance '$NOTEBOOK_INSTANCE_NAME' creation initiated."

while true; do
  STATUS=$(aws sagemaker describe-notebook-instance --notebook-instance-name $NOTEBOOK_INSTANCE_NAME \
            --query 'NotebookInstanceStatus' --output text --region $REGION)
  echo "Notebook instance status: $STATUS"
  if [ "$STATUS" == "InService" ]; then
    break
  elif [ "$STATUS" == "Failed" ]; then
    echo "Notebook instance creation failed."
    exit 1
  fi
  sleep 10
done

# Get the notebook instance URL
INSTANCE_URL=$(aws sagemaker describe-notebook-instance --notebook-instance-name $NOTEBOOK_INSTANCE_NAME \
               --query 'Url' --output text --region $REGION)
echo "Notebook instance '$NOTEBOOK_INSTANCE_NAME' is ready and accessible at: https://$INSTANCE_URL"