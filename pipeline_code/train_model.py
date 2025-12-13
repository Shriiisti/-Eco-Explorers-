import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms as T

# --- Configuration (MAX EFFICIENCY & HIGH QUALITY) ---
UNIFIED_JSON_PATH = 'data_sources/unified_segmentation_data.json'
PROCESSED_MASKS_DIR = 'processed_masks' 
BATCH_SIZE = 4
NUM_EPOCHS = 5          # 5 epochs for optimal fine-tuning (best quality)
LEARNING_RATE = 0.0001
TARGET_SIZE = 512       
MAX_SAMPLES = 4000      # Training on 4000 samples for high quality

# --- Dataset Class ---
class SolarPanelDataset(Dataset):
    def __init__(self, json_path, transform=None, max_samples=None):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Unified JSON file not found at: {json_path}")

        with open(json_path, 'r') as f:
            data_list = json.load(f)
        
        self.data_list = data_list[:max_samples] if max_samples else data_list
        print(f"Using a dataset of {len(self.data_list)} samples for training.")
        
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_point = self.data_list[idx]
        img_path = data_point['image_path']
        
        mask_filename = os.path.basename(data_point['mask_path'])
        mask_path = os.path.join(PROCESSED_MASKS_DIR, mask_filename) 
        
        # Read Image (BGR to RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read Mask (Grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at: {mask_path}")
        
        mask = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        image_pil = T.ToPILImage()(image)
        
        image_tensor = self.transform(image_pil)
        
        mask_tensor = T.Resize((TARGET_SIZE, TARGET_SIZE), 
                               interpolation=T.InterpolationMode.NEAREST)(mask_tensor)

        return image_tensor, mask_tensor
    

# --- Training Function ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = T.Compose([
        T.Resize((TARGET_SIZE, TARGET_SIZE)),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = SolarPanelDataset(json_path=UNIFIED_JSON_PATH, transform=train_transform, max_samples=MAX_SAMPLES)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"CRITICAL DATA LOADING ERROR: {e}")
        return

    model_save_path = 'solar_panel_segmentation_model.pth'
    
    # Initialize the DeepLabV3 model with PRE-TRAINED WEIGHTS (BEST MODEL)
    print("Loading DeepLabV3 with pre-trained COCO/ImageNet weights...")
    # NOTE: The 'DEFAULT' setting automatically sets aux_loss=True for training.
    model = deeplabv3_resnet50(weights='DEFAULT') 
    
    # Reconfigure the final classification layer for a single binary class (our solar panel)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) 
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print(f"Starting EFFICIENT training for {NUM_EPOCHS} epoch(s) on {TARGET_SIZE}x{TARGET_SIZE}...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            
            # The output during training is a dictionary when aux_loss is active, but we only use the main 'out'
            outputs = model(images)['out'] 
            
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {epoch_loss:.4f}")

    print("\nTraining complete!")
    
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# --- Execution ---
if __name__ == "__main__":
    train_model()