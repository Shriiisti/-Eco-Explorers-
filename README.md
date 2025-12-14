Solar Panel Segmentation Model:

Project Overview:
This repository contains the codebase, trained model, and results for a machine learning pipeline designed for solar panel segmentation. The goal is to accurately identify and delineate individual solar panels within satellite or aerial imagery.
├──artefacts
|   ├── processed_masks
├── data_sources 
|   ├──raw_annotations
|   ├── raw_images
|   ├── unified_segmentations_data.json
├── pipeline_code               # Scripts for data preprocessing, training, and inference
│   ├── data_loader.py
│   ├── train_model.py
│   └── inference.py
├── trained_model_file        # Contains the final trained model weights
│   └── solar_panel_segmentation_model.pth 
├── model_card                 # Documentation on the model's design and performance
│   └── model_card.md 
├──prediction files


Setup Instructions:
1. Cloning the Repository
   this project uses Git LFS (Large File Storage) to handle the large model file and cloned to avoid the file corruption.
   a)Install Git LFS:
     git lfs install
   b)Clone the Repository:
     git clone https://github.com/Shriiisti/-Eco-Explorers-/tree/master
2. Environment Setup
Created and activated a Python virtual environment, then installed the required dependencies.
   a)Create Environment:
     python -m venv
   b)Activating Environment:
     .\venv\Scripts\Activate
3.Install Dependencies:
    pip install torch torchvision numpy pandas matplotlib scikit-learn opencv-python Pillow 

Running the Pipeline:
1. Training:
   To retrain the model from scratch.
   python pipeline_code/train_model.py 
2. Evaluation:
  evaluated the existing trained model on a test dataset.
  python pipeline_code/inference.py --model_path trained_model_file/solar_panel_segmentation_model.pth 


Results
Training Results
MetricEpoch     1Epoch   3Final  (Epoch 5)
Training Loss   0.1428   0.0699    0.0671
Inferred mIoU   0.55      0.72       0.75


 Contact:
 Developed by ECO EXPLORERS

GitHub: https://github.com/Shriiisti/-Eco-Explorers-/tree/master
name:Shristi C C
     Kushala Laxman Gouda
     
Email: shristi.chengappac@gmail.com
       kushalagouda0818@gmail.com
