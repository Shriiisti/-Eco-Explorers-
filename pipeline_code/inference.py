import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms as T
import cv2
import numpy as np
import os
from PIL import Image

# --- Configuration (MUST MATCH train_model.py and Submission Structure) ---
# NOTE: Model path updated to reflect the 'Trained model file' submission folder.
MODEL_PATH = 'trained_model_file/solar_panel_segmentation_model.pth' 
# Set your test image path here (use one from your 'data_sources' folder)
IMAGE_TO_TEST = 'data_sources/raw_images/sp0013_png.rf.6f9af39b059a490dfab91a3c1b747ac0.jpg' 
TARGET_SIZE = 512 
OUTPUT_DIR = 'Prediction files' # The required output folder

# --- Model Definition (Includes All Critical Loading Fixes) ---
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")

    # CRITICAL FIX 1: aux_loss=True must be set to match the saved model structure 
    # (because it was trained with weights='DEFAULT').
    model = deeplabv3_resnet50(weights=None, aux_loss=True) 
    
    # Re-configure the final classification layer for 1 output class (Solar Panel)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) 
    
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None, None
    
    model.to(device)
    
    # CRITICAL FIX 2: Force model and Batch Normalization layers to eval() mode for stable inference.
    model.eval() 
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            
    return model, device

# --- Image Preprocessing ---
def preprocess_image(image_path, target_size):
    preprocess = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_pil = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image_pil).unsqueeze(0)
    # Read the original image with OpenCV for visualization (BGR format)
    original_cv2 = cv2.imread(image_path)
    
    return input_tensor, original_cv2

# --- Visualization and SAVING Function (Saves Output for Submission) ---
def visualize_and_save_result(original_img, prediction_mask, image_path): 
    H, W, _ = original_img.shape
    # Resize the prediction mask back to the original image dimensions
    prediction_resized = cv2.resize(
        prediction_mask.astype(np.uint8) * 255, 
        (W, H), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create the green mask overlay
    colored_mask = np.zeros_like(original_img, dtype=np.uint8)
    colored_mask[prediction_resized > 0] = [0, 255, 0] # Green in BGR (OpenCV format)
    
    # Blend the original image and the mask (60% original, 40% mask)
    blended_img = cv2.addWeighted(original_img, 0.6, colored_mask, 0.4, 0)
    
    # --- FILE SAVING CODE ---
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Creates 'Prediction files' if it doesn't exist
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Save the final green overlay image
    overlay_save_path = os.path.join(OUTPUT_DIR, f'{base_filename}_overlay.png')
    cv2.imwrite(overlay_save_path, blended_img)
    print(f"Saved final green overlay image to: {overlay_save_path}")
    
    # 2. Save the raw binary mask
    mask_save_path = os.path.join(OUTPUT_DIR, f'{base_filename}_mask.png')
    cv2.imwrite(mask_save_path, prediction_resized)
    print(f"Saved raw binary mask to: {mask_save_path}")
    
    # Display the result (Optional: for immediate visual confirmation)
    cv2.imshow('Segmentation Mask (Green Overlay)', blended_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# --- Main Inference Loop ---
def run_inference():
    model, device = load_model(MODEL_PATH)
    
    if model is None or device is None:
        print("Inference failed: Could not load model.")
        return

    # 1. Preprocess the image
    print(f"Testing on image: {IMAGE_TO_TEST}")
    if not os.path.exists(IMAGE_TO_TEST):
        print(f"Error: Test image not found at {IMAGE_TO_TEST}. Please check the path.")
        return
        
    input_tensor, original_cv2 = preprocess_image(IMAGE_TO_TEST, TARGET_SIZE)
    input_tensor = input_tensor.to(device)

    # 2. Run Inference
    with torch.no_grad():
        # The output dictionary contains the main output under the 'out' key
        output = model(input_tensor)['out'] 

    # 3. Post-process the output
    # Convert logits to probabilities using sigmoid
    probabilities = torch.sigmoid(output).cpu().squeeze().numpy()
    
    # Use the proven aggressive threshold (0.1) for revealing accurate predictions
    threshold = 0.1
    print(f"Applying segmentation threshold of: {threshold}")
    prediction_mask = (probabilities > threshold).astype(np.bool_)

    # 4. Visualize and Save the result
    visualize_and_save_result(original_cv2, prediction_mask, IMAGE_TO_TEST)
    
if __name__ == "__main__":
    run_inference()