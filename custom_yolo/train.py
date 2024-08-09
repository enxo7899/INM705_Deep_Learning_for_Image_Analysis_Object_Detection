import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import numpy as np
import wandb
import torch.nn.functional as F

# Initialize WandB for logging and tracking experiments
wandb.init(project="YOLO-From_Scratch", entity="enxo7899")

# PASCAL VOC class labels
PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom transformation to resize images and adjust bounding boxes
class ResizeWithBbox:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes):
        original_size = image.size
        image = transforms.functional.resize(image, self.size)

        scale_x = self.size[0] / original_size[0]
        scale_y = self.size[1] / original_size[1]

        transformed_boxes = []
        for box in boxes:
            cls_id, x_center, y_center, width, height = box
            x_center = x_center * scale_x
            y_center = y_center * scale_y
            width = width * scale_x
            height = height * scale_y
            transformed_boxes.append([cls_id, x_center, y_center, width, height])

        return image, transformed_boxes

# YOLODataset class to load the dataset, apply transformations, and encode targets
class YOLODataset(Dataset):
    def __init__(self, annotations_dir, images_dir, transform=None, grid_size=7, num_boxes=2, num_classes=20):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.transform = transform
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        annotation_files = os.listdir(self.annotations_dir)
        annotations = []
        for file in annotation_files:
            file_path = os.path.join(self.annotations_dir, file)
            image_name = file.replace('.txt', '.jpg')
            with open(file_path, 'r') as f:
                boxes = [line.strip().split() for line in f.readlines()]
                annotations.append((image_name, boxes))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_name, boxes = self.annotations[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        boxes = [[float(x) for x in box] for box in boxes]

        if self.transform:
            image, boxes = self.transform(image, boxes)

        target = self.encode_target(boxes)

        return transforms.ToTensor()(image), target

    def encode_target(self, boxes):
        target = torch.zeros((self.grid_size, self.grid_size, self.num_boxes * 5 + self.num_classes))
        for box in boxes:
            cls_id = int(box[0])
            x_center, y_center, width, height = box[1], box[2], box[3], box[4]

            grid_x = min(int(x_center * self.grid_size), self.grid_size - 1)
            grid_y = min(int(y_center * self.grid_size), self.grid_size - 1)

            x_center_grid = x_center * self.grid_size - grid_x
            y_center_grid = y_center * self.grid_size - grid_y
            width_grid = width * self.grid_size
            height_grid = height * self.grid_size

            if target[grid_y, grid_x, 0] == 0:
                target[grid_y, grid_x, 0:5] = torch.tensor([1, x_center_grid, y_center_grid, width_grid, height_grid])
                target[grid_y, grid_x, 5 + cls_id] = 1

        return target

# Initialize the dataset with the custom transform
transform = ResizeWithBbox((448, 448))

# Path to the annotations and images
annotations_dir = '/content/pascalvoc-yolo/labels'
images_dir = '/content/pascalvoc-yolo/images'
dataset = YOLODataset(annotations_dir, images_dir, transform=transform)

# Dataset Splitting
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
def collate_fn(batch):
    images = []
    targets = []

    for img, box in batch:
        images.append(img)
        targets.append(box)

    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)

    return images, targets

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

# Define the YOLO-Like model architecture
class FScratchYOLO(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super(FScratchYOLO, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Convolutional layers with CSP-like blocks
        self.focus = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.csp1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.csp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.csp3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.csp4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        # Fully connected layers for final predictions
        test_input = torch.randn(1, 3, 448, 448)
        test_output = self.forward_features(test_input)
        print(f"Output shape after conv layers: {test_output.shape}")

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(test_output.numel(), 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, grid_size * grid_size * (num_boxes * 5 + num_classes))
        )

    def forward_features(self, x):
        x = self.focus(x)
        x = checkpoint(self.csp1, x)
        x = checkpoint(self.csp2, x)
        x = checkpoint(self.csp3, x)
        x = checkpoint(self.csp4, x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.grid_size, self.grid_size, self.num_boxes * 5 + self.num_classes)
        return x

# YOLO loss function to calculate loss during training
class YOLOLoss(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20, lambda_coord=2, lambda_noobj=0.05):
        super(YOLOLoss, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        pred_boxes = predictions[..., :10].reshape(batch_size, self.grid_size, self.grid_size, self.num_boxes, 5)
        pred_conf = pred_boxes[..., 0]
        pred_xywh = pred_boxes[..., 1:]
        pred_class_probs = predictions[..., 10:]

        target_boxes = targets[..., :10].reshape(batch_size, self.grid_size, self.grid_size, self.num_boxes, 5)
        target_conf = target_boxes[..., 0]
        target_xywh = target_boxes[..., 1:]
        target_class_probs = targets[..., 10:]

        iou_scores = self.compute_iou(pred_xywh, target_xywh)

        coord_loss = self.lambda_coord * F.mse_loss(pred_xywh, target_xywh, reduction='sum')

        conf_loss = F.mse_loss(pred_conf, target_conf, reduction='sum') + \
                    self.lambda_noobj * F.mse_loss(pred_conf * (1 - target_conf), target_conf * (1 - pred_conf), reduction='sum')

        class_loss = F.mse_loss(pred_class_probs, target_class_probs, reduction='sum')

        total_loss = coord_loss + conf_loss + class_loss
        return total_loss / batch_size

    def compute_iou(self, pred_boxes, target_boxes):
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        return iou

# Metric calculation functions for precision, recall, F1-score, IoU, and mAP
def calculate_metrics(predictions, targets, threshold=0.5):
    preds_class = (predictions[..., 10:] > threshold).reshape(-1, model.num_classes).cpu().numpy()
    targets_class = targets[..., 10:].reshape(-1, model.num_classes).cpu().numpy()

    precision = precision_score(targets_class.argmax(axis=1), preds_class.argmax(axis=1), average='macro', zero_division=1)
    recall = recall_score(targets_class.argmax(axis=1), preds_class.argmax(axis=1), average='macro', zero_division=1)
    f1 = f1_score(targets_class.argmax(axis=1), preds_class.argmax(axis=1), average='macro', zero_division=1)
    iou = calculate_iou(predictions[..., :4], targets[..., :4]).mean().item()
    mAP = calculate_map(predictions, targets)

    return precision, recall, f1, iou, mAP

# Calculate IoU for the predicted and target bounding boxes
def calculate_iou(pred, target):
    pred_x1 = pred[..., 0] - pred[..., 2] / 2
    pred_y1 = pred[..., 1] - pred[..., 3] / 2
    pred_x2 = pred[..., 0] + pred[..., 2] / 2
    pred_y2 = pred[..., 1] + pred[..., 3] / 2

    target_x1 = target[..., 0] - target[..., 2] / 2
    target_y1 = target[..., 1] - target[..., 3] / 2
    target_x2 = target[..., 0] + target[..., 2] / 2
    target_y2 = target[..., 1] + target[..., 3] / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou

# Calculate mAP (mean Average Precision) for the predictions and targets
def calculate_map(predictions, targets, iou_threshold=0.5):
    aps = []
    for i in range(predictions.size(0)):
        pred_boxes = predictions[i][..., :4].reshape(-1, 4)
        target_boxes = targets[i][..., :4].reshape(-1, 4)

        pred_conf = predictions[i][..., 4].reshape(-1)
        target_conf = targets[i][..., 0].reshape(-1)

        sorted_indices = torch.argsort(pred_conf, descending=True)
        tp = torch.zeros(len(pred_conf))
        fp = torch.zeros(len(pred_conf))

        for j, idx in enumerate(sorted_indices):
            if target_conf[idx] > 0 and calculate_iou(pred_boxes[idx], target_boxes[idx]) > iou_threshold:
                tp[j] = 1
            else:
                fp[j] = 1

        tp = torch.cumsum(tp, dim=0)
        fp = torch.cumsum(fp, dim=0)
        precisions = tp / (tp + fp + 1e-6)
        recalls = tp / (len(target_boxes[target_conf > 0]) + 1e-6)
        ap = torch.sum((recalls[1:] - recalls[:-1]) * precisions[1:]).item()
        aps.append(ap)

    mAP = np.mean(aps)
    return mAP

# Plot confusion matrix to visualize model performance
def plot_confusion_matrix(predictions, targets, threshold=0.5):
    preds_class = (predictions[..., 10:] > threshold).reshape(-1, model.num_classes).cpu().numpy()
    targets_class = targets[..., 10:].reshape(-1, model.num_classes).cpu().numpy()
    conf_matrix = confusion_matrix(targets_class.argmax(axis=1), preds_class.argmax(axis=1))

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=PASCAL_CLASSES, yticklabels=PASCAL_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Initialize model, loss function, and optimizer
model = FScratchYOLO().to(DEVICE)
loss_fn = YOLOLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Learning rate scheduler for dynamic adjustment of the learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Gradient scaler for mixed precision training
scaler = GradScaler()

# Training loop with gradient accumulation and mixed precision
num_epochs = 10
accumulation_steps = 8

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        with autocast():
            predictions = model(images)
            loss = loss_fn(predictions, targets) / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    elapsed_time = time.time() - start_time

    # Adjust learning rate
    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            with autocast():
                predictions = model(images)
                loss = loss_fn(predictions, targets)
            val_loss += loss.item()

            all_predictions.append(predictions)
            all_targets.append(targets)

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        precision, recall, f1, iou, mAP = calculate_metrics(all_predictions, all_targets)
        plot_confusion_matrix(all_predictions, all_targets)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Total Loss: {avg_loss:.4f}, "
          f"Validation Loss: {val_loss / len(val_loader):.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, "
          f"IoU: {iou:.4f}, mAP: {mAP:.4f}, "
          f"FPS: {len(train_loader.dataset) / elapsed_time:.2f}")

    wandb.log({
        "epoch": epoch + 1,
        "loss": avg_loss,
        "val_loss": val_loss / len(val_loader),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "mAP": mAP,
        "fps": len(train_loader.dataset) / elapsed_time
    })

    torch.save(model.state_dict(), f'Yolov3_epoch{epoch+1}.pth')

wandb.finish()

print("Training complete.")
