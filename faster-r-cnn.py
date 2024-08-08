import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.ops as ops
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
import wandb

# Initialize WandB
api_key = "9ce954fd827fd8d839648cb3708ff788ad51bafa"
wandb.login(key=api_key)
wandb.init(project="faster-rcnn-project-fromscratch", entity="enxo7899")

# Custom Dataset Class for Pascal VOC
class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.annotations['class'] = self.label_encoder.fit_transform(self.annotations['class'])

        # Balance the dataset by limiting the 'person' class to 5000 images
        self.balance_dataset()

    def balance_dataset(self):
        person_class = self.label_encoder.transform(['person'])[0]
        person_indices = self.annotations[self.annotations['class'] == person_class].index
        if len(person_indices) > 5000:
            # Randomly sample 5000 indices from the person class
            person_indices = np.random.choice(person_indices, 5000, replace=False)
        non_person_indices = self.annotations[self.annotations['class'] != person_class].index
        # Combine person and non-person indices
        balanced_indices = np.concatenate((person_indices, non_person_indices))
        # Update annotations to only include the balanced set
        self.annotations = self.annotations.loc[balanced_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        boxes = self.annotations.iloc[idx, 4:].values.astype("float").reshape(-1, 4)
        labels = self.annotations.iloc[idx, 1].astype("int")

        sample = {"image": image, "boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((400, 400)),  # Resize images to a fixed size
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Custom collate function to handle batches with variable-length elements
def custom_collate(batch):
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {'image': torch.stack(images), 'boxes': boxes, 'labels': labels}

# Initialize Dataset
dataset = PascalVOCDataset(root_dir="VOCdevkit/VOC2012/JPEGImages", annotations_file="pascal_voc_annotations.csv", transform=transform)

# Initialize DataLoader with custom collate function using the balanced dataset
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate)

# Check a sample batch
sample_batch = next(iter(dataloader))
print("Sample Batch Images Shape:", sample_batch["image"].shape)
print("Sample Batch Boxes:", sample_batch["boxes"])
print("Sample Batch Labels:", sample_batch["labels"])

# Define a more complex ResNet-like backbone from scratch
class ComplexBackbone(nn.Module):
    def __init__(self):
        super(ComplexBackbone, self).__init__()
        self.layer1 = self._make_layer(3, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.layer4 = self._make_layer(256, 512, 2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        for _ in range(1, num_blocks):
            layers.extend([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# Define the Region Proposal Network (RPN)
class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        super(RegionProposalNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(self.conv(x))
        rpn_logits = self.cls_logits(x)
        rpn_bbox_pred = self.bbox_pred(x)
        return rpn_logits, rpn_bbox_pred

# Generate dummy proposals
def generate_dummy_proposals(feature_map_size, num_proposals=9):
    proposals = []
    for _ in range(num_proposals):
        x1 = torch.randint(0, feature_map_size // 2, (1,)).item()
        y1 = torch.randint(0, feature_map_size // 2, (1,)).item()
        x2 = x1 + torch.randint(16, 64, (1,)).item()  # Random width
        y2 = y1 + torch.randint(16, 64, (1,)).item()  # Random height
        proposals.append([x1, y1, x2, y2])
    return torch.tensor(proposals, dtype=torch.float32)

# Define RoI Pooling Layer
class RoIPooling(nn.Module):
    def __init__(self, output_size):
        super(RoIPooling, self).__init__()
        self.output_size = output_size

    def forward(self, features, proposals, image_shapes):
        return ops.roi_pool(features, proposals, self.output_size, spatial_scale=1.0)

# Define the Classification and Regression Heads
class ClassificationAndRegressionHeads(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassificationAndRegressionHeads, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred

# Assemble the complete Faster R-CNN model
class FasterRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_pooling, heads):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_pooling = roi_pooling
        self.heads = heads

    def forward(self, images, proposals, image_shapes):
        # Extract features using the backbone
        features = self.backbone(images)
        # Generate proposals using the RPN
        rpn_logits, rpn_bbox_pred = self.rpn(features)
        # Perform RoI Pooling
        pooled_features = self.roi_pooling(features, proposals, image_shapes)
        # Classify and refine boxes
        cls_logits, bbox_pred = self.heads(pooled_features)
        return cls_logits, bbox_pred, rpn_logits, rpn_bbox_pred

# Define the loss functions
classification_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()

# Move model to device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model components with a complex backbone
backbone = ComplexBackbone()
rpn = RegionProposalNetwork(in_channels=512)  
roi_pooling = RoIPooling(output_size=(7, 7))
heads = ClassificationAndRegressionHeads(in_features=7*7*512, num_classes=21)

# Instantiate the complete model
model = FasterRCNN(backbone, rpn, roi_pooling, heads).to(device)

# Define optimizer with learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  

# Training parameters
num_epochs = 2  
log_interval = 100  

# Custom IoU calculation
def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    if len(box1) == 0 or len(box2) == 0:
        return np.array([0.0])

    # Ensure boxes are numpy arrays
    box1 = np.array(box1)
    box2 = np.array(box2)

    # Handle single box input
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    # Determine the coordinates of the intersection rectangle
    inter_x1 = np.maximum(box1[:, 0], box2[:, 0])
    inter_y1 = np.maximum(box1[:, 1], box2[:, 1])
    inter_x2 = np.minimum(box1[:, 2], box2[:, 2])
    inter_y2 = np.minimum(box1[:, 3], box2[:, 3])

    # Compute the area of intersection rectangle
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Compute the area of union
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / np.maximum(union_area, 1e-8)

    return iou

# Function to calculate mean Average Precision (mAP)
def calculate_map(true_boxes, pred_boxes, iou_threshold=0.5):
    """Calculates the mean Average Precision (mAP) over a dataset."""
    all_precisions = []

    for tb, pb in zip(true_boxes, pred_boxes):
        if len(pb) == 0:
            all_precisions.append(0)
            continue

        ious = calculate_iou(tb, pb)
        sorted_indices = np.argsort(-ious)
        ious = ious[sorted_indices]

        tp = ious >= iou_threshold
        fp = ious < iou_threshold

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / (len(tb) + 1e-8)

        precision = np.concatenate(([1.0], precision, [0.0]))
        recall = np.concatenate(([0.0], recall, [1.0]))

        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = np.maximum(precision[i - 1], precision[i])

        indices = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
        all_precisions.append(ap)

    return np.mean(all_precisions) if all_precisions else 0.0

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    rpn_cls_loss_epoch = 0.0
    rpn_reg_loss_epoch = 0.0
    roi_cls_loss_epoch = 0.0
    roi_reg_loss_epoch = 0.0

    start_time = time.time()

    all_true_labels = []
    all_pred_labels = []
    all_true_boxes = []
    all_pred_boxes = []

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        targets = [{'boxes': boxes.to(device), 'labels': labels.to(device)} for boxes, labels in zip(batch['boxes'], batch['labels'])]

        # Image shapes should match the original shapes before resizing
        image_shapes = [(img.size(1), img.size(2)) for img in images]  # (height, width) format

        # Generate proposals for each image in the batch
        proposals = [generate_dummy_proposals(image_shapes[i][0]).to(device) for i in range(len(images))]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        cls_logits, bbox_pred, rpn_logits, rpn_bbox_pred = model(images, proposals, image_shapes)

        # Calculate losses for each image
        rpn_cls_loss = 0
        rpn_reg_loss = 0
        roi_cls_loss = 0
        roi_reg_loss = 0

        for i in range(len(images)):
            # Calculate RPN classification loss
            rpn_logits_flat = rpn_logits[i].permute(1, 2, 0).reshape(-1, 2)
            rpn_targets = torch.zeros(rpn_logits_flat.shape[0], dtype=torch.long, device=device)  

            rpn_cls_loss += classification_loss_fn(rpn_logits_flat, rpn_targets)

            # Calculate RPN regression loss (using dummy targets here)
            rpn_bbox_pred_flat = rpn_bbox_pred[i].permute(1, 2, 0).reshape(-1, 4)
            rpn_reg_loss += bbox_loss_fn(rpn_bbox_pred_flat, torch.zeros_like(rpn_bbox_pred_flat, device=device))

            # Calculate ROI losses
            roi_cls_loss += classification_loss_fn(cls_logits[i].view(-1, 21), targets[i]['labels'].view(-1))
            roi_reg_loss += bbox_loss_fn(bbox_pred[i].view(-1, 4), targets[i]['boxes'].view(-1, 4))

            # Collect predictions for metrics
            all_true_labels.append(targets[i]['labels'].item())
            all_pred_labels.append(torch.argmax(cls_logits[i]).item())

            # Collect true and predicted boxes for IoU and mAP calculation
            all_true_boxes.append(targets[i]['boxes'].cpu().numpy())
            all_pred_boxes.append(bbox_pred[i].detach().cpu().numpy())

        # Total loss
        loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate losses
        epoch_loss += loss.item()
        rpn_cls_loss_epoch += rpn_cls_loss.item()
        rpn_reg_loss_epoch += rpn_reg_loss.item()
        roi_cls_loss_epoch += roi_cls_loss.item()
        roi_reg_loss_epoch += roi_reg_loss.item()

        # Log batch info
        if (batch_idx + 1) % log_interval == 0:
            print(f"Batch [{batch_idx + 1}/{len(dataloader)}], "
                  f"Loss: {loss.item():.4f}, "
                  f"RPN Cls Loss: {rpn_cls_loss.item():.4f}, "
                  f"RPN Reg Loss: {rpn_reg_loss.item():.4f}, "
                  f"ROI Cls Loss: {roi_cls_loss.item():.4f}, "
                  f"ROI Reg Loss: {roi_reg_loss.item():.4f}")

    # Calculate metrics
    precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=1)
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=1)
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    iou = np.mean([np.max(calculate_iou(tb, pb)) for tb, pb in zip(all_true_boxes, all_pred_boxes) if len(tb) > 0 and len(pb) > 0])
    map_score = calculate_map(all_true_boxes, all_pred_boxes)

    fps = len(dataloader.dataset) / (time.time() - start_time)  # Frames Per Second

    # Print epoch loss and metrics
    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss/len(dataloader):.4f}")
    print(f"  RPN Classification Loss: {rpn_cls_loss_epoch/len(dataloader):.4f}")
    print(f"  RPN Regression Loss: {rpn_reg_loss_epoch/len(dataloader):.4f}")
    print(f"  ROI Classification Loss: {roi_cls_loss_epoch/len(dataloader):.4f}")
    print(f"  ROI Regression Loss: {roi_reg_loss_epoch/len(dataloader):.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, FPS: {fps:.2f}")
    print(f"  IoU: {iou:.4f}, mAP: {map_score:.4f}")

    # Log metrics to WandB
    wandb.log({
        "Epoch": epoch + 1,
        "Total Loss": epoch_loss / len(dataloader),
        "RPN Classification Loss": rpn_cls_loss_epoch / len(dataloader),
        "RPN Regression Loss": rpn_reg_loss_epoch / len(dataloader),
        "ROI Classification Loss": roi_cls_loss_epoch / len(dataloader),
        "ROI Regression Loss": roi_reg_loss_epoch / len(dataloader),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "IoU": iou,
        "mAP": map_score,
        "FPS": fps
    })

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.label_encoder.classes_, yticklabels=dataset.label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Update learning rate
    scheduler.step()

    # Save model checkpoint
    torch.save(model.state_dict(), f'faster_rcnn_epoch_{epoch+1}.pth')

print("Training complete.")

# Finish the WandB run
wandb.finish()
