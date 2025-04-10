import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import tensorflow as tf
import cv2
from tkinter import Tk, Label, Button, filedialog, Frame
from PIL import Image, ImageTk

# ----------- CNN Class Definition (matching .pth structure) -----------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ----------- Load Models -----------
AUDIO_MODEL_PATH = "best_sound.keras"
FOOTPRINT_MODEL_PATH = "best_footprint.pth"

audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
footprint_model = CNN(num_classes=10).to(device)
footprint_model.load_state_dict(torch.load(FOOTPRINT_MODEL_PATH, map_location=device))
footprint_model.eval()

# ----------- Labels -----------
audio_labels = {
   0: 'bear', 1: 'cat', 2: 'cow', 3: 'dog', 4: 'donkey',
   5: 'elephant', 6: 'horse', 7: 'lion', 8: 'monkey', 9: 'sheep'
}

footprint_labels = {
   0: 'bear', 1: 'bobcat', 2: 'deer', 3: 'fox', 4: 'horse',
   5: 'lion', 6: 'mouse', 7: 'racoon', 8: 'squirrel', 9: 'wolf'
}

safe_animals_audio = {'cat', 'cow', 'dog', 'donkey', 'horse', 'sheep'}
unsafe_animals_audio = {'bear', 'elephant', 'lion', 'monkey'}

safe_animals_footprint = {'deer', 'horse', 'mouse', 'squirrel', 'racoon'}
unsafe_animals_footprint = {'bear', 'bobcat', 'fox', 'lion', 'wolf'}

# ----------- Processing Functions -----------
def extract_mel_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y, _ = librosa.effects.trim(y, top_db=20)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=256, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec + 1e-9, ref=np.max)
        mel_spec_norm = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-9)
        padded = np.zeros((128, 174))
        padded[:, :mel_spec_norm.shape[1]] = mel_spec_norm[:, :174]
        return np.expand_dims(padded, axis=(0, -1))
    except Exception as e:
        print(f"Audio error: {e}")
        return None

def preprocess_footprint_image(file_path):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        print(f"Image error: {e}")
        return None

# ----------- Prediction Handlers -----------
def predict_animal():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if not file_path:
        return
    features = extract_mel_spectrogram(file_path)
    if features is None:
        result_label.config(text="Error processing audio.")
        return
    predictions = audio_model.predict(features)[0]
    idx = np.argmax(predictions)
    animal = audio_labels[idx]
    confidence = predictions[idx] * 100
    status = "Safe" if animal in safe_animals_audio else "Unsafe"
    result_label.config(text=f"Detected: {animal} ({status})\nConfidence: {confidence:.2f}%")
    display_image(audio_image_label, "images", animal)

def predict_footprint():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if not file_path:
        return
    tensor = preprocess_footprint_image(file_path)
    if tensor is None:
        footprint_result_label.config(text="Error processing image.")
        return
    with torch.no_grad():
        preds = footprint_model(tensor)
        idx = torch.argmax(preds).item()
        animal = footprint_labels[idx]
        confidence = torch.softmax(preds, dim=1)[0, idx].item() * 100
        status = "Safe" if animal in safe_animals_footprint else "Unsafe"
        footprint_result_label.config(text=f"Detected: {animal} ({status})\nConfidence: {confidence:.2f}%")
        display_image(footprint_image_label, "animals", animal)

def display_image(label_widget, folder, animal):
    path = os.path.join(folder, animal)
    for ext in ['.jpg', '.jpeg', '.png']:
        full_path = path + ext
        if os.path.exists(full_path):
            img = Image.open(full_path).resize((250, 200))
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo, text="")
            label_widget.image = photo
            return
    label_widget.config(text="Image not found", image="")

# ----------- GUI Setup -----------
root = Tk()
root.title("Animal Classifier")
root.geometry("400x500")

home_frame = Frame(root)
Label(home_frame, text="TRACK & TRACE").pack(pady=20)
Button(home_frame, text="Classify Audio", command=lambda: show_frame(audio_frame)).pack(pady=5)
Button(home_frame, text="Classify Footprint", command=lambda: show_frame(footprint_frame)).pack(pady=5)
home_frame.pack()

audio_frame = Frame(root)
Label(audio_frame, text="Upload Animal Sound").pack(pady=10)
Button(audio_frame, text="Upload Audio", command=predict_animal).pack()
Button(audio_frame, text="Back", command=lambda: show_frame(home_frame)).pack(pady=5)
result_label = Label(audio_frame, text="")
result_label.pack(pady=10)
audio_image_label = Label(audio_frame)
audio_image_label.pack()

footprint_frame = Frame(root)
Label(footprint_frame, text="Upload Animal Footprint").pack(pady=10)
Button(footprint_frame, text="Upload Image", command=predict_footprint).pack()
Button(footprint_frame, text="Back", command=lambda: show_frame(home_frame)).pack(pady=5)
footprint_result_label = Label(footprint_frame, text="")
footprint_result_label.pack(pady=10)
footprint_image_label = Label(footprint_frame)
footprint_image_label.pack()

def show_frame(frame):
    for f in [home_frame, audio_frame, footprint_frame]:
        f.pack_forget()
    frame.pack()

root.mainloop()
