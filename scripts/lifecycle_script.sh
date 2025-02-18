#!/bin/bash
set -e

cd /home/ec2-user/SageMaker
git clone https://github.com/alejomongua/yolo_aws.git
chown -R ec2-user:ec2-user yolo_aws
cd yolo_aws
