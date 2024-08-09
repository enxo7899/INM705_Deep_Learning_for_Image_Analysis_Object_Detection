import torch
import torchvision.transforms as transforms
import torchvision.ops as ops
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model = FasterRCNN(backbone, rpn, roi_pooling, heads).to(device)
model.load_state_dict(torch.load('path/name.pth'))
model.eval()  # Set the model to evaluation mode

# Define a transformation for the input image
transform = transforms.Compose([
    transforms.Resize((400, 400)),  
    transforms.ToTensor(),          
])

# Define a function to perform inference on a single image
def perform_inference(image_path, model, device, threshold=0.5):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transformed_image = transform(image).unsqueeze(0).to(device)

    # Generate dummy proposals for the image
    proposals = [generate_dummy_proposals(feature_map_size=50).to(device)]  

    # Perform the forward pass
    with torch.no_grad():
        cls_logits, bbox_pred, _, _ = model(transformed_image, proposals, [(400, 400)])

    # Convert outputs to numpy arrays
    scores = torch.nn.functional.softmax(cls_logits, dim=-1).squeeze(0).cpu().numpy()
    bboxes = bbox_pred.view(-1, 4).cpu().numpy()

    # Ensure predictions are aligned
    if scores.shape[0] != bboxes.shape[0]:
        num_preds = min(scores.shape[0], bboxes.shape[0])
        scores = scores[:num_preds]
        bboxes = bboxes[:num_preds]

    # Filter predictions based on the threshold
    pred_labels = np.argmax(scores, axis=1)
    pred_scores = np.max(scores, axis=1)
    keep_indices = pred_scores > threshold
    pred_boxes = bboxes[keep_indices]
    pred_labels = pred_labels[keep_indices]
    pred_scores = pred_scores[keep_indices]

    # Visualize predictions
    visualize_predictions(image, pred_boxes, pred_labels, pred_scores)

# Define a function to visualize predictions
def visualize_predictions(image, boxes, labels, scores):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for box, label, score in zip(boxes, labels, scores):
        if len(box) == 4:  # Ensure box has 4 coordinates
            xmin, ymin, xmax, ymax = box
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2))
            ax.text(xmin, ymin - 2, f'{dataset.label_encoder.inverse_transform([label])[0]}: {score:.3f}',
                    bbox=dict(facecolor='blue', alpha=0.5), fontsize=10, color='white')

    plt.axis('off')
    plt.show()

# Perform inference on a given image
image_path = '/content/VOCdevkit/VOC2012/JPEGImages/2007_000061.jpg'  # Replace with your image path
perform_inference(image_path, model, device)
