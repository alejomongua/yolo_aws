import argparse
import os
import tarfile
import random
import shutil
from glob import glob
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

# VOC label mapping (adjust if needed)
VOC_LABELS = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
    'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
    'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}


def extract_tar(tar_path, extract_path):
    print(f"Extracting {tar_path} to {extract_path} ...")
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete.")


def split_dataset(image_files, train_ratio=0.7, val_ratio=0.2, seed=42):
    random.seed(seed)
    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return image_files[:train_end], image_files[train_end:val_end], image_files[val_end:]


def process_sample(image_path, annot_path, target_size):
    """
    Process a single image and its annotation:
      - Load and resize image.
      - Parse XML annotation to extract objects.
      - Convert bounding boxes to normalized YOLO format:
          x_center, y_center, width, height (all normalized by target_size) and class id.
    Returns:
      image_np: The processed image as a NumPy array.
      boxes: A NumPy array of shape (N, 5) where each row is [x_center, y_center, width, height, class_id].
    """
    # Load image
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        orig_width, orig_height = img.size
        img = img.resize((target_size, target_size))
        image_np = np.array(img)

    # Parse annotation XML
    tree = ET.parse(annot_path)
    root = tree.getroot()
    boxes = []
    # Use the original size provided in the XML if available; fallback to image dimensions.
    size_elem = root.find('size')
    if size_elem is not None:
        orig_width = float(size_elem.find('width').text)
        orig_height = float(size_elem.find('height').text)

    for obj in root.findall('object'):
        label = obj.find('name').text
        class_id = VOC_LABELS.get(label, -1)
        if class_id == -1:
            continue  # Skip unknown labels
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        # Convert coordinates to the resized image scale
        new_xmin = xmin * target_size / orig_width
        new_ymin = ymin * target_size / orig_height
        new_xmax = xmax * target_size / orig_width
        new_ymax = ymax * target_size / orig_height
        # Compute center and dimensions (normalized between 0 and 1)
        x_center = ((new_xmin + new_xmax) / 2.0) / target_size
        y_center = ((new_ymin + new_ymax) / 2.0) / target_size
        width = (new_xmax - new_xmin) / target_size
        height = (new_ymax - new_ymin) / target_size
        boxes.append([x_center, y_center, width, height, class_id])

    boxes = np.array(boxes, dtype=np.float32) if boxes else np.empty(
        (0, 5), dtype=np.float32)
    return image_np, boxes


def process_and_save_files(image_files, voc_root, dest_dir, target_size):
    """
    For each image in image_files, process the image and corresponding annotation.
    Save the result as a compressed .npz file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    annots_dir = os.path.join(voc_root, "Annotations")

    for img_path in image_files:
        base = os.path.basename(img_path)
        name, _ = os.path.splitext(base)
        annot_path = os.path.join(annots_dir, f"{name}.xml")
        if not os.path.exists(annot_path):
            print(
                f"Warning: Annotation for {name} not found. Skipping sample.")
            continue
        try:
            image_np, boxes = process_sample(img_path, annot_path, target_size)
            out_path = os.path.join(dest_dir, f"{name}.npz")
            np.savez_compressed(out_path, image=image_np, boxes=boxes)
        except Exception as e:
            print(f"Error processing {name}: {e}")
    print(f"Processed samples saved to {dest_dir}")


def main(args):
    # Directories provided by SageMaker Processing job
    input_dir = args.input_dir   # e.g., /opt/ml/processing/input
    output_dir = args.output_dir  # e.g., /opt/ml/processing/output

    # Find the tar file
    tar_files = glob(os.path.join(input_dir, "*.tar*"))
    if not tar_files:
        raise ValueError("No tar file found in the input directory.")
    tar_path = tar_files[0]

    # Create a temporary extraction directory
    extract_path = os.path.join("/tmp", "voc_extracted")
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)

    extract_tar(tar_path, extract_path)

    # Locate the VOC folder; we assume the tar contains a "VOCdevkit" folder
    vocdevkit_dir = os.path.join(extract_path, "VOCdevkit")
    if not os.path.exists(vocdevkit_dir):
        raise ValueError("VOCdevkit folder not found after extraction.")

    # Assume we work with VOC2012
    voc_root = os.path.join(vocdevkit_dir, "VOC2012")
    if not os.path.exists(voc_root):
        raise ValueError("VOC2012 folder not found in VOCdevkit.")

    # Get list of image files
    images_dir = os.path.join(voc_root, "JPEGImages")
    image_files = glob(os.path.join(images_dir, "*.jpg"))
    if not image_files:
        raise ValueError("No images found in the JPEGImages folder.")

    # Split dataset into train, validation, and test sets.
    train_files, val_files, test_files = split_dataset(image_files,
                                                       train_ratio=args.train_ratio,
                                                       val_ratio=args.val_ratio)
    print(f"Total images: {len(image_files)}")
    print(
        f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # For each split, process the images and annotations and save as .npz files.
    for split_name, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        dest_dir = os.path.join(output_dir, split_name)
        print(f"Processing {split_name} split...")
        process_and_save_files(files, voc_root, dest_dir, args.image_size)

    print("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/input",
                        help="Input directory where the tar file is located")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output",
                        help="Output directory to store processed data")
    parser.add_argument("--train_ratio", type=float,
                        default=0.7, help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float,
                        default=0.2, help="Ratio of validation data")
    parser.add_argument("--image_size", type=int, default=448,
                        help="Target size for resizing images")
    args = parser.parse_args()
    main(args)
