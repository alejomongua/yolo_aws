#!/usr/bin/env python
# train.py
import argparse
import glob
import os
import torch

from model import YOLO
from loss import YOLOLoss
from data import get_datasets, get_data_loaders
from train_utils import train_model, evaluate_map
from train_config import NUMBER_OF_CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float,
                        default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float,
                        default=0.0005, help="Weight decay")
    parser.add_argument("--momentum", type=float,
                        default=0.90, help="Momentum")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory where dataset is stored")
    parser.add_argument("--model-dir", type=str, default=".",
                        help="Directory to save the trained model")
    parser.add_argument("--load-weights", action="store_true",
                        help="Flag to load existing weights")
    parser.add_argument("--force-train", action="store_true",
                        help="Flag to force training even if weights exist")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets and data loaders (assumes data has been preprocessed if desired)
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, val_dataset, test_dataset)

    # Check for existing weights in the model directory
    weights = None
    modelos = sorted(glob.glob(os.path.join(
        args.model_dir, 'model_*.pth')), key=os.path.getmtime)
    if len(modelos) > 0:
        weights = modelos[-1]

    # Initialize model and load weights if available
    model = YOLO(num_classes=NUMBER_OF_CLASSES)
    if weights is not None:
        print(f'Loading weights from {weights}')
        state_dict = torch.load(weights)
        model.load_state_dict(state_dict)

    # Define loss function
    loss_function = YOLOLoss()

    # Train the model
    model = train_model(model, train_loader, val_loader,
                        num_epochs=args.epochs, loss_function=loss_function)
    model.eval()

    # Evaluate the model on the test dataset
    print("Evaluating model on test dataset...")
    evaluate_map(model, test_loader, iou_thresh=0.5, conf_thresh=0.5)


if __name__ == '__main__':
    main()
