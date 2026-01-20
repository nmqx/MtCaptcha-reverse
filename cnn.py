import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import glob
import sys

# --- CONFIGURATION ---
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)} # 0 réservé pour le "blank"
IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

IMG_HEIGHT = 50
IMG_WIDTH = 200
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Configuration: Utilisation de {DEVICE} ---")

# --- 1. UTILITAIRES (Encodage/Décodage & Nettoyage) ---

def encode_text(text):
    """Convertit une chaine de caractères en liste d'indices."""
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]

def decode_text(preds):
    """Décodage Greedy CTC : Argmax -> Suppression doublons -> Suppression blanks."""
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

def clean_label(raw_label):
    """Retire tout caractère non autorisé du nom de fichier."""
    return "".join([c for c in raw_label if c in CHAR2IDX])

# --- 2. DATASET (Chargement robuste) ---

class CaptchaDataset(Dataset):
    def __init__(self, dir_path):
        self.image_paths = glob.glob(os.path.join(dir_path, "*.gif"))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)
        
        try:
            # Extraction et nettoyage du label
            raw_label = filename.split('_')[-1].replace('.gif', '')
            label_text = clean_label(raw_label)
            
            if len(label_text) == 0:
                return None # Label vide ou invalide -> on ignore

            # Chargement image : Grayscale -> Resize -> Normalize
            image = Image.open(path).convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
            image = np.array(image).astype("float32") / 255.0
            image = np.expand_dims(image, 0) # Ajout dimension channel (1, H, W)
            
            return torch.tensor(image), label_text
        except Exception:
            return None # Fichier corrompu -> on ignore

def robust_collate_fn(batch):
    """Gère les None renvoyés par __getitem__ pour ne pas faire planter le DataLoader."""
    batch = [item for item in batch if item is not None]
    if not batch: return None
    
    images, texts = zip(*batch)
    images = torch.stack(images)
    
    # Préparation des cibles pour CTC Loss (format 1D concaténé)
    targets = []
    target_lengths = []
    for t in texts:
        encoded = encode_text(t)
        targets.extend(encoded)
        target_lengths.append(len(encoded))
    
    return images, torch.tensor(targets, dtype=torch.long), torch.tensor(target_lengths, dtype=torch.long), texts

# Chargement des DataLoaders
train_loader = None
test_loader = None

if os.path.exists("captured_hits"):
    full_dataset = CaptchaDataset("captured_hits")
    if len(full_dataset) > 0:
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=robust_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=robust_collate_fn)
        print(f"--- Données chargées : {len(train_dataset)} train, {len(test_dataset)} test ---")

# --- 3. MODELE (CRNN avec Initialisation) ---
# 

class CRNN(nn.Module):
    def __init__(self, img_height, num_classes):
        super(CRNN, self).__init__()
        
        # --- CNN (Extraction de features) ---
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        
        # Calcul de la hauteur après 2 poolings (50 -> 25 -> 12)
        self.feature_height = img_height // 4
        self.projection = nn.Linear(64 * self.feature_height, 64)
        
        # --- RNN (Séquence temporelle) ---
        self.rnn = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.fc_out = nn.Linear(256, num_classes + 1)
        
        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier/Kaiming pour éviter que le loss ne stagne."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # 1. Features visuelles
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # 2. Reshape pour RNN : (Batch, Channel, H, W) -> (Batch, W, Features)
        x = x.permute(0, 3, 1, 2) 
        batch, width, channel, height = x.size()
        x = x.reshape(batch, width, channel * height)
        
        # 3. Séquence
        x = self.relu(self.projection(x))
        x, _ = self.rnn(x)
        x = self.fc_out(x)
        
        return F.log_softmax(x, dim=2)

model = CRNN(IMG_HEIGHT, NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# Chargement des poids existants
if os.path.exists("crnn_weights.pt"):
    try:
        model.load_state_dict(torch.load("crnn_weights.pt", map_location=DEVICE))
        print("--- Poids chargés depuis crnn_weights.pt ---")
    except:
        print("--- Erreur chargement poids, initialisation à neuf ---")

# --- 4. ENTRAINEMENT ---

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    print(f"Démarrage entrainement pour {epoch} époques...")
    
    for i in range(epoch):
        total_loss = 0
        num_batches = 0
        
        for batch_data in train_loader:
            if batch_data is None: continue # Skip batch corrompu
            
            images, targets, target_lengths, _ = batch_data
            images, targets = images.to(device), targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(images) 
            preds = preds.permute(1, 0, 2) # (Time, Batch, Class) requis par CTCLoss
            
            input_lengths = torch.full(size=(images.size(0),), fill_value=50, dtype=torch.long).to(device)
            
            loss = ctc_loss(preds, targets, input_lengths, target_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            
            # CLIP GRADIENT (Important pour éviter le crash du loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {i + 1} - Loss: {avg_loss:.4f}")

# Lance l'entrainement si on a des données et qu'on le souhaite (ou si pas de poids)
if train_loader and (not os.path.exists("crnn_weights.pt") or "FORCE_TRAIN" in os.environ):
    train(model, DEVICE, train_loader, optimizer, EPOCHS)
    torch.save(model.state_dict(), "crnn_weights.pt")
    print("--- Modèle sauvegardé ---")

# --- 5. TEST / EVALUATION ---

def test(model, device, test_loader):
    model.eval()
    print("\n--- Évaluation sur le jeu de test ---")
    with torch.no_grad():
        # On prend juste le premier batch pour afficher quelques exemples
        data_iter = iter(test_loader)
        batch = next(data_iter)
        if batch is None: return

        images, _, _, real_texts = batch
        images = images.to(device)
        
        outputs = model(images)
        predictions = decode_text(outputs)
        
        print(f"{'RÉEL':<20} | {'PRÉDIT':<20}")
        print("-" * 45)
        for i in range(min(5, len(images))):
            print(f"{real_texts[i]:<20} | {predictions[i]:<20}")

if test_loader:
    test(model, DEVICE, test_loader)

# --- 6. PREDICTION INDIVIDUELLE ---

def predict_image(image_path):
    if not os.path.exists(image_path):
        return None

    try:
        image = Image.open(image_path).convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
        image_arr = np.array(image).astype("float32") / 255.0
        image_tensor = torch.tensor(image_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            text = decode_text(output)
            return text[0]
    except Exception as e:
        print(f"Erreur prédiction: {e}")
        return None

# Exemple d'appel final
print("\n--- Test manuel ---")
result = predict_image("test/lKgZIZ.gif")
if result:
    print(f"Prédiction pour l'image : {result}")
else:
    print("Image introuvable ou erreur.")