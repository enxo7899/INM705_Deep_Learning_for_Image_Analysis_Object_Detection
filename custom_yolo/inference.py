import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set device to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PASCAL VOC class labels
PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

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

# Function to load the model and set it to evaluation mode
def load_model(model_path):
    model = FScratchYOLO().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# Function to preprocess the image and return it along with the original image
def preprocess_image(image_path, img_size=448):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image_tensor, image

# Function to draw bounding boxes on the image
def draw_bboxes(image, predictions, threshold=0.5):
    grid_size = predictions.shape[1]
    img_size = image.size[0]  # Assuming image is square
    cell_size = img_size / grid_size

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i in range(grid_size):
        for j in range(grid_size):
            for b in range(2):  # Two bounding boxes per grid cell
                box = predictions[0, i, j, b*5:(b*5)+5].cpu().detach().numpy()
                conf = box[0]
                if conf > threshold:
                    x_center = (box[1] + i) * cell_size
                    y_center = (box[2] + j) * cell_size
                    width = box[3] * img_size
                    height = box[4] * img_size
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2

                    # Draw the bounding box
                    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                    # Add class label and confidence score
                    class_probs = predictions[0, i, j, 10:].cpu().detach().numpy()
                    class_id = np.argmax(class_probs)
                    label = PASCAL_CLASSES[class_id]
                    plt.text(x_min, y_min, f'{label}: {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

# Function for running inference on a single image
def run_inference(model, image_path):
    image_tensor, original_image = preprocess_image(image_path)
    with torch.no_grad():
        predictions = model(image_tensor)

    draw_bboxes(original_image, predictions)

# Example usage:
model_path = 'Yolov3_epoch1.pth'
image_path = '/content/pascalvoc-yolo/images/000007.jpg'  # Update with the path to your test image

# Load the model
model = load_model(model_path)

# Run inference
run_inference(model, image_path)
