import numpy as np
import json
import os
import cv2
from tqdm import tqdm 
import sys
from skimage.draw import polygon # Added for reliable drawing logic

# --- Configuration ---
RAW_ANNOTATIONS_DIR = 'data_sources/raw_annotations' 
RAW_IMAGE_DIR = 'data_sources/raw_images' 
PROCESSED_MASKS_DIR = 'processed_masks' 
OUTPUT_UNIFIED_FILE = 'data_sources/unified_segmentation_data.json'
TARGET_SIZE = 512 # <<< OPTIMIZED FOR SPEED (was 768)

# --- Helper Functions ---

def normalize_filename(fname):
    """Aggressively cleans a filename to find its core ID."""
    name = fname.lower()
    name = os.path.splitext(name)[0]
    if '.rf.' in name:
        name = name.split('.rf.')[0]
    if name.endswith('_jpg'):
        name = name[:-4]
    elif name.endswith('_png'):
        name = name[:-4]
    elif name.endswith('_jpeg'):
        name = name[:-5]
    return name

def create_mask(width, height, polygons, target_size):
    """
    Creates a combined binary mask from all polygons using skimage.draw.polygon.
    This is the most reliable method for COCO format.
    """
    # 1. Create original size mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 2. Draw all polygons
    for seg_list in polygons:
        if isinstance(seg_list, list) and len(seg_list) >= 6 and (len(seg_list) % 2) == 0:
            try:
                # Polygons are flat: [x1, y1, x2, y2, ...]
                r = seg_list[1::2] # rows (y)
                c = seg_list[0::2] # columns (x)

                rr, cc = polygon(r, c, shape=(height, width))
                mask[rr, cc] = 255 
            except Exception:
                continue

    # 3. Resize the final mask to the TARGET_SIZE (512x512)
    mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return mask_resized

# --- Main Processing Function ---

def run_data_processor():
    os.makedirs(PROCESSED_MASKS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_UNIFIED_FILE), exist_ok=True)
    
    print("Indexing raw images...")
    image_map = {} 
    
    try:
        raw_files = os.listdir(RAW_IMAGE_DIR) 
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Folder not found at path: {RAW_IMAGE_DIR}")
        return

    for f in raw_files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            norm_name = normalize_filename(f)
            image_map[norm_name] = os.path.join(RAW_IMAGE_DIR, f)
            
    print(f"Indexed {len(image_map)} images from disk.")

    try:
        annotation_files_list = os.listdir(RAW_ANNOTATIONS_DIR)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Folder not found at path: {RAW_ANNOTATIONS_DIR}")
        return
        
    annotation_files = [f for f in annotation_files_list if f.lower().endswith('.json') and f.lower().count('.json') == 1]
    
    print(f"Found {len(annotation_files)} annotation files. Processing...")
    
    all_annotations = {} 
    image_meta = {}
    
    for filename in tqdm(annotation_files, desc="Reading JSONs"):
        try:
            with open(os.path.join(RAW_ANNOTATIONS_DIR, filename), 'r') as f:
                data = json.load(f)
        except Exception as e: 
            print(f"\nError reading {filename}: {e}")
            continue
            
        if 'images' in data:
            for img in data['images']:
                img_id = str(img.get('id'))
                image_meta[img_id] = {'w': img.get('width'), 'h': img.get('height'), 'fname': img.get('file_name', '')}
        
        if 'annotations' in data:
            for anno in data['annotations']:
                img_id = str(anno.get('image_id'))
                seg = anno.get('segmentation')
                if seg and isinstance(seg, list):
                    if img_id not in all_annotations: all_annotations[img_id] = []
                    all_annotations[img_id].extend(seg)

    unified_data = []
    count_success = 0
    
    for img_id, polygons in tqdm(all_annotations.items(), desc="Generating Masks"):
        meta = image_meta.get(img_id)
        if not meta: continue

        json_filename = meta['fname']
        norm_json_name = normalize_filename(json_filename)
        real_path = image_map.get(norm_json_name)
        
        if not real_path: continue
            
        raw_image_path = real_path
        
        try:
            img = cv2.imread(raw_image_path)
            if img is None:
                print(f"\nSkipping: Failed to load image at path: {raw_image_path}")
                continue
                
            w = meta.get('w', TARGET_SIZE)
            h = meta.get('h', TARGET_SIZE)
            
            # CALLING THE NEW, RELIABLE MASK CREATION FUNCTION
            mask = create_mask(w, h, polygons, TARGET_SIZE) 
            
            mask_fname = f"{norm_json_name}_mask.png"
            mask_path = os.path.join(PROCESSED_MASKS_DIR, mask_fname)
            cv2.imwrite(mask_path, mask)
            
            unified_data.append({
                "image_id": norm_json_name,
                "image_path": raw_image_path,
                "mask_path": mask_path
            })
            count_success += 1
            
        except Exception as e:
            print(f"\nCRITICAL I/O ERROR: Failed to process file {raw_image_path}. Error: {e}")
            continue

    with open(OUTPUT_UNIFIED_FILE, 'w') as f:
        json.dump(unified_data, f, indent=4)
        
    print(f"\nFINAL RESULT: Successfully created {count_success} masks at {TARGET_SIZE}x{TARGET_SIZE}.")

if __name__ == "__main__":
    run_data_processor()