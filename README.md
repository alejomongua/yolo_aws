# AWS ML Specialist YOLO Project

This repository contains an implementation of the YOLO object detection algorithm using PyTorch, integrated with AWS SageMaker for a complete ML workflow. The project was developed as part of preparation for the AWS ML Specialist certification and includes end-to-end pipelines for data preprocessing, model training (with debugging enabled), and evaluation.

## Overview

- **YOLO Implementation:** A custom YOLO network modeled after the original paper.
- **Data Handling:** Preprocessing of the VOC Pascal dataset using SageMaker Processing jobs.
- **Training:** Model training using SageMaker Training jobs with integrated Debugger for detailed monitoring.
- **Modular Codebase:** The project is split into modular components for better maintainability:
  - `model.py` – Contains the YOLO model definition.
  - `loss.py` – Custom loss functions.
  - `data.py` – Data preparation, dataset loaders, and collate functions.
  - `utils.py` – Utility functions for dataset extraction, visualization, and evaluation.
  - `train.py` – Main entry point for training the model.

## Directory Structure

aws-ml-specialist-yolo/
 ├── data.py # Data loading and preprocessing utilities
 ├── loss.py # YOLO loss function implementation
 ├── model.py # YOLO model architecture
 ├── train.py # Training job entry point
 ├── utils.py # Utility functions (extraction, visualization, evaluation)
 ├── preprocessing/ # Contains the preprocessing script (preprocess.py) and related files
 │   └── preprocess.py # Script to extract, split, and convert the VOC dataset
 └── README.md # This file

## Prerequisites

- **AWS Account:** With SageMaker permissions.

## Setup and Usage

### 1. Create an IAM account with SageMaker permissions

- **Create an IAM Role:** Create an IAM role with SageMaker permissions (e.g., `AmazonSageMakerFullAccess`).
- **Configure AWS CLI:** Use the `aws configure` command to set up your AWS CLI with the IAM user credentials.

### 2. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/alejomongua/yolo_aws
cd yolo_aws
```

### 3. Create and launch a SageMaker Notebook instance

```bash
bash deploy_resources.sh
```

### 4. Run the provided notebook

The deployment script shows the URL of the notebook instance. Open the notebook and run the cells to preprocess the dataset, train the model, and evaluate the results. The notebook contains this flow:

#### 1. Data Preprocessing

Use the provided SageMaker Processing job to preprocess the dataset.

- **Preprocessing Script:** Located in `preprocessing/preprocess.py`
- **Job Execution:** Run the notebook cell that launches the processing job. The script will:
  - Extract the VOC tar file.
  - Split the dataset into training, validation, and test sets.
  - Convert the dataset into a convenient format for training.
  - Save the processed data to S3.

#### 2. Model Training with SageMaker Debugger

The training job uses the `train.py` script along with the other modules.

- **Debugger Integration:** The training job is configured with SageMaker Debugger to capture metrics (e.g., loss values) during training.
- **Launching the Training Job:** In the notebook, configure the PyTorch estimator by specifying the `entry_point` (`train.py`), `source_dir` (the training code directory), and input channels for the preprocessed data.
- **Hyperparameters:** Customize hyperparameters (e.g., epochs, batch size, learning rate) via command-line arguments or directly in the estimator configuration.


## AWS Services Used

- **SageMaker Processing:** For data extraction and preprocessing.
- **SageMaker Training:** For model training with integrated debugging.
- **S3:** For dataset storage and as a data source for training.
- **SageMaker Debugger:** To monitor training and capture debug information.

## References

- [SageMaker Processing Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [SageMaker Debugger Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger.html)
- [AWS SageMaker Script Mode](https://docs.aws.amazon.com/sagemaker/latest/dg/script-mode.html)

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.
