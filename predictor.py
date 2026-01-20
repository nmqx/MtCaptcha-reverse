import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

# --- Constants & Config ---
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)} # 0 for "blank"
IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

IMG_HEIGHT = 50
IMG_WIDTH = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helpers ---

def decode_text(preds):
    """CTC Decoding"""
    pred_indices = torch.argmax(preds, dim=2).detach().cpu().numpy()
    text_results = []
    for sequence in pred_indices:
        decoded = []
        prev_char = -1
        for char_idx in sequence:
            if char_idx != prev_char and char_idx != 0:
                decoded.append(IDX2CHAR[char_idx])
            prev_char = char_idx
        text_results.append("".join(decoded))
    return text_results

# --- Model Definition ---

class CRNN(nn.Module):
    def __init__(self, img_height, num_classes):
        super(CRNN, self).__init__()
        
        # --- CNN ---
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        
        # Height calculations: 50 -> 25 -> 12
        self.feature_height = img_height // 4
        self.projection = nn.Linear(64 * self.feature_height, 64)
        
        # --- RNN ---
        self.rnn = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.fc_out = nn.Linear(256, num_classes + 1)

    def forward(self, x):
        # 1. Visual Features
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # 2. Reshape for RNN : (Batch, Channel, H, W) -> (Batch, W, Features)
        x = x.permute(0, 3, 1, 2) 
        batch, width, channel, height = x.size()
        x = x.reshape(batch, width, channel * height)
        
        # 3. Sequence
        x = self.relu(self.projection(x))
        x, _ = self.rnn(x)
        x = self.fc_out(x)
        
        return F.log_softmax(x, dim=2)

# --- Initialization ---

model = CRNN(IMG_HEIGHT, NUM_CLASSES).to(DEVICE)

# Load weights if available
weights_path = os.path.join(os.path.dirname(__file__), "crnn_weights.pt")
if os.path.exists(weights_path):
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading weights: {e}")
else:
    print(f"Warning: Weights file not found at {weights_path}")

model.eval()

# --- Prediction Function ---

def cnn(image_path):
    """
    Predicts the text of the captcha image at the given path.
    Returns the predicted string or None if failed.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    try:
        # Load and Preprocess
        image = Image.open(image_path).convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
        image_arr = np.array(image).astype("float32") / 255.0
        # Add dimensions: (1, 1, H, W) for Batch and Channel
        image_tensor = torch.tensor(image_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image_tensor)
            text_list = decode_text(output)
            return text_list[0]
            
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None
