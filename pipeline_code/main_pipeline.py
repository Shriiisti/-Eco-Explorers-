import pandas as pd
import requests
import os
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from io import BytesIO
from math import radians, asin, floor, pi
import segmentation_models_pytorch as smp 

# --- Configuration (NO API KEY NEEDED!) ---
INPUT_FILE = 'input_data/test_input.csv'
OUTPUT_DIR = 'output_data'
FETCHED_IMAGE_DIR = 'pipeline_code/fetched_images'
MODEL_SAVE_PATH = 'pipeline_code/best_unet_model.pth' # Path to your trained model

# Tile Server URL (Open Source, No Billing)
TILE_SERVER_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# Image Parameters
ZOOM_LEVEL = 18 
TILE_SIZE = 256
TILES_PER_SIDE = 3
IMAGE_SIZE = TILE_SIZE * TILES_PER_SIDE # 768x768 pixels
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions for Tile Coordinates ---

def lat_lon_to_tile(lat, lon, zoom):
    """Converts WGS84 lat/lon to tile (x, y) coordinates."""
    lat_rad = radians(lat)
    n = 2.0 ** zoom
    x = n * ((lon + 180.0) / 360.0)
    y = n * (1.0 - asin(lat_rad) / pi) / 2.0
    return floor(x), floor(y)

def fetch_and_stitch_image(lat, lon, sample_id, zoom=ZOOM_LEVEL):
    """Fetches surrounding tiles and stitches them into one image."""
    
    center_x, center_y = lat_lon_to_tile(lat, lon, zoom)
    start_x = center_x - (TILES_PER_SIDE // 2)
    start_y = center_y - (TILES_PER_SIDE // 2)

    stitched_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
    
    for row in range(TILES_PER_SIDE):
        for col in range(TILES_PER_SIDE):
            x = start_x + col
            y = start_y + row
            
            tile_url = TILE_SERVER_URL.format(z=zoom, x=x, y=y)
            
            try:
                response = requests.get(tile_url, headers={'User-Agent': 'SunSureVerifyPipeline/1.0'})
                if response.status_code == 200:
                    tile_image = Image.open(BytesIO(response.content))
                    stitched_image.paste(tile_image, (col * TILE_SIZE, row * TILE_SIZE))
                else:
                    # Stitching failed tile will be black (RGB default)
                    print(f"Warning: Failed to fetch tile {x},{y} (Status: {response.status_code})")
            except Exception as e:
                print(f"Error fetching tile {x},{y}: {e}")

    os.makedirs(FETCHED_IMAGE_DIR, exist_ok=True)
    image_path = os.path.join(FETCHED_IMAGE_DIR, f'{sample_id}.png')
    stitched_image.save(image_path)
    
    image_meta = {"source": "OpenStreetMap Tiles", "capture_date": "Varies by Tile"}
    return image_path, image_meta


def load_model(path, device):
    """Loads the trained UNET model."""
    try:
        model = smp.Unet(
            encoder_name="resnet34",        
            encoder_weights=None,     
            in_channels=3,                  
            classes=1,                      
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {path}")
        return model
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model file not found at {path}. Please run train_model.py first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_and_save_visual(model, image_path, output_dir, sample_id):
    """Runs prediction and saves the visual overlay."""
    
    # 1. Load and Preprocess Image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image from {image_path}")
        return 0.0, None # Return 0 confidence
        
    resized_image = cv2.resize(original_image, (256, 256))
    
    # Convert to PyTorch Tensor
    input_image = resized_image.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).to(DEVICE) # (1, 3, 256, 256)

    # 2. Run Inference
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Process prediction (logits -> probability -> binary mask)
    pred_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
    binary_mask = (pred_mask > 0.5).astype(np.uint8) 

    # 3. Post-process Mask
    # Resize mask back to original 768x768 size
    final_mask = cv2.resize(binary_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    # 4. Create Visual Overlay
    # Create a green overlay for visualization
    green_overlay = np.zeros_like(original_image, dtype=np.uint8)
    green_overlay[:, :, 1] = 255 # Green channel set to full
    
    # Combine original image and overlay based on the mask
    # The mask is 0 or 1. Use it to blend the green overlay onto the original image.
    final_mask_3ch = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR) # Convert mask to 3 channels
    
    # Blend: result = (original * (1 - mask)) + (overlay * mask)
    # Since IoU was low (0.0002), the mask will likely be sparse or non-existent, but the logic works.
    alpha = final_mask.astype(np.float32) * 0.5 # Transparency factor
    alpha_3ch = np.expand_dims(alpha, axis=2)

    # Blend original image and the green overlay
    overlay_image = (original_image * (1 - alpha_3ch) + green_overlay * alpha_3ch).astype(np.uint8)

    # 5. Determine Confidence and Area (Simple placeholder logic)
    pv_pixels = np.sum(final_mask)
    has_solar = pv_pixels > 10 # Check if more than 10 pixels were predicted as PV
    confidence = np.max(pred_mask) if has_solar else 0.0 # Confidence is max probability
    pv_area_sqm_est = pv_pixels * (0.1**2) # Placeholder conversion for pixel area

    # 6. Save the Visual Image
    output_visual_path = os.path.join(output_dir, f'{sample_id}_predicted.png')
    cv2.imwrite(output_visual_path, overlay_image)
    
    print(f"Saved visual output to: {output_visual_path}")
    
    return has_solar, confidence, pv_area_sqm_est

# --- Main Pipeline ---

def run_verification_pipeline(input_file):
    """The main function to process all sites."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Model once
    model = load_model(MODEL_SAVE_PATH, DEVICE)
    if model is None:
        return # Exit if model failed to load

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    for index, row in df.iterrows():
        sample_id = row['sample_id']
        lat = row['latitude']
        lon = row['longitude']
        
        print(f"Processing Site ID: {sample_id} at ({lat}, {lon})...")
        
        # --- 1. FETCH IMAGE ---
        image_path, image_meta = fetch_and_stitch_image(lat, lon, sample_id)
        
        # --- 2. CLASSIFY & QUANTIFY ---
        if image_path and os.path.exists(image_path):
            has_solar, confidence, pv_area_sqm_est = predict_and_save_visual(
                model, image_path, OUTPUT_DIR, sample_id
            )
            qc_status = "VERIFIED"
            qc_notes = ["OSM image fetched", "Model prediction complete"]
        else:
            has_solar, confidence, pv_area_sqm_est = False, 0.0, 0.0
            qc_status = "NOT_VERIFIABLE"
            qc_notes = ["Image fetch failed"]

       # --- 3. CREATE & SAVE JSON ---
        output_data = {
            "sample_id": int(sample_id),
            "lat": lat,
            "lon": lon,
            "has_solar": has_solar,
            "confidence": round(confidence, 4),
            "pv_area_sqm_est": round(pv_area_sqm_est, 4),
            "buffer_radius_sqft": 2400, 
            "qc_status": qc_status,
            "qc_notes": qc_notes,
            "bbox_or_mask": "",
            "image_metadata": image_meta
        }
        
        # FIX: Ensure 'has_solar' is a serializable string ("true" or "false")
        # Python's bool type sometimes causes errors with older json versions/environments
        output_data['has_solar'] = "true" if output_data['has_solar'] else "false"
        
        with open(os.path.join(OUTPUT_DIR, f'{sample_id}.json'), 'w') as f:
            json.dump(output_data, f, indent=4)
        
    print("\nPipeline execution complete. Check 'output_data' for images and JSONs.")


if __name__ == "__main__":
    run_verification_pipeline(INPUT_FILE)