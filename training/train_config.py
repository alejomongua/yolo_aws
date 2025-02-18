# train_config.py

NUMBER_OF_CLASSES = 20  # VOC has 20 classes
GRID_SIZE = 7           # 7x7 grid
NUMBER_OF_BBOXES = 1     # One bounding box per cell
IMAGE_SIZE = 448

# Optimization hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.90
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Loss hyperparameters
LAMBDA_COORD = 50
LAMBDA_NOOB = 0.5
LAMBDA_OBJ = 10
LAMBDA_CLASS = 1

# Detection thresholds
CONF_THRESH = 0.5
IOU_THRESH = 0.5

# VOC labels
VOC_LABELS = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
    'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
    'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

# Derived constants
CLASS_LABELS = list(VOC_LABELS.keys())
