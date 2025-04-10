import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
import os

# Parameters
img_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "C:\\Users\\anisa\\AppData\\Local\\Programs\\PROJECTS\\colab\\best_footprint.pth"
images_folder = "C:\\Users\\anisa\\AppData\\Local\\Programs\\PROJECTS\\colab\\animals"
class_labels = ['bear', 'bobcat', 'deer', 'fox', 'horse', 'lion', 'mouse', 'racoon', 'squirrel', 'wolf']

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# CNN Model (same as training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (img_size//8) * (img_size//8), 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, len(class_labels))
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# GUI classify function
def classify_footprint():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Load and predict footprint image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        result_label.config(text="Error: image not found.")
        return

    img_resized = cv2.resize(img, (img_size, img_size))
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        predicted_idx = torch.argmax(probs).item()
        confidence = probs[0][predicted_idx].item()

    label = class_labels[predicted_idx]
    result_label.config(text=f"Detected: {label.capitalize()}\nConfidence: {confidence:.2%}")

    # Show the corresponding animal image
    animal_img_path = os.path.join(images_folder, f"{label}.jpg")  # or .png
    if os.path.exists(animal_img_path):
        animal_img = Image.open(animal_img_path).resize((200, 200))
        tk_animal_img = ImageTk.PhotoImage(animal_img)
        animal_img_label.config(image=tk_animal_img)
        animal_img_label.image = tk_animal_img
    else:
        animal_img_label.config(text="No image for this animal", image='')

# GUI Layout
root = Tk()
root.title("Footprint Classifier")
root.geometry("350x400")

Label(root, text="Upload a Footprint Image").pack(pady=10)
Button(root, text="Upload Image", command=classify_footprint).pack(pady=5)

# Animal image preview only
animal_img_label = Label(root)
animal_img_label.pack(pady=10)

# Prediction result
result_label = Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
