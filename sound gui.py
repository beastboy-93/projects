import os
import numpy as np
import librosa
import tensorflow as tf
from tkinter import Tk, Label, Button, filedialog, Frame
from PIL import Image, ImageTk

# Load Model
MODEL_PATH = "best_sound.keras"  # Ensure model is in the same directory
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Ensure 'best_model.keras' is in the same directory.")
model = tf.keras.models.load_model(MODEL_PATH)

# Correct Label Order from Training Report
labels = {
    0: 'bear', 1: 'cat', 2: 'cow', 3: 'dog', 4: 'donkey', 5: 'elephant', 6: 'horse', 7: 'lion', 8: 'monkey', 9: 'sheep'
}

safe_animals = {'cat', 'cow', 'dog', 'donkey', 'horse', 'sheep'}
unsafe_animals = {'bear', 'elephant', 'lion', 'monkey'}

# Extract Mel Spectrogram
def extract_mel_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y, _ = librosa.effects.trim(y, top_db=20)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=256, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec + 1e-9, ref=np.max)
        mel_spec_norm = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-9)
        mel_spec_padded = np.zeros((128, 174))
        mel_spec_padded[:, :min(174, mel_spec_norm.shape[1])] = mel_spec_norm[:, :174]
        return np.expand_dims(mel_spec_padded, axis=(0, -1))
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# GUI
def start_app():
    home_frame.pack_forget()
    app_frame.pack()

def predict_animal():
    file_path = filedialog.askopenfilename(filetypes=[('Audio Files', '*.wav')])
    if not file_path:
        return
    spectrogram = extract_mel_spectrogram(file_path)
    if spectrogram is not None:
        prediction = model.predict(spectrogram)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        animal = labels.get(predicted_label, "Unknown")
        print(f"Predicted Label Index: {predicted_label}, Animal: {animal}, Confidence: {confidence:.2f}%")
        result_text = f"Predicted Animal: {animal.capitalize()} (Confidence: {confidence:.2f}%)\n{'Safe' if animal in safe_animals else 'Unsafe'} Animal"
        result_label.config(text=result_text)
        
        # Display Image
        image_path = f"images/{animal}"
        for ext in ['.jpg', '.jpeg', '.png']:
            if os.path.isfile(image_path + ext):
                image = Image.open(image_path + ext).resize((250, 200))
                photo = ImageTk.PhotoImage(image)
                image_label.config(image=photo)
                image_label.image = photo
                return
        image_label.config(text='Image not found', image='')

# Main Window
root = Tk()
root.title("Animal Sound Classifier")
root.geometry("400x500")

# Home Page
home_frame = Frame(root)
Label(home_frame, text="Welcome to Animal Sound Classifier", font=("Helvetica", 14)).pack(pady=20)
Button(home_frame, text="Start", command=start_app).pack(pady=10)
home_frame.pack()

# Main App
app_frame = Frame(root)
Label(app_frame, text="Upload Animal Sound", font=("Helvetica", 14)).pack(pady=20)
Button(app_frame, text="Upload Audio", command=predict_animal).pack(pady=10)
result_label = Label(app_frame, text="", font=("Helvetica", 12))
result_label.pack(pady=10)
image_label = Label(app_frame)
image_label.pack(pady=20)

root.mainloop()
