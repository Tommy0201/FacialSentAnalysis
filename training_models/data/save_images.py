import os
import random
import pandas as pd
from PIL import Image
import io
from datasets import load_dataset

def create_emotion_directories(base_dir):
    """Create directories for each emotion if they don't exist."""
    emotion_labels = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"
    }
    
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories for each emotion
    directories = {}
    for label, emotion in emotion_labels.items():
        path = os.path.join(base_dir, f"{label}_{emotion}")
        os.makedirs(path, exist_ok=True)
        directories[label] = path
        
    return directories

def save_sample_images(data, output_dir, samples_per_class=10, seed=42):
    """
    Save sample images from each emotion class.
    
    Args:
        data (pd.DataFrame): DataFrame containing 'label' and 'image' columns
        output_dir (str): Base directory to save images
        samples_per_class (int): Number of samples to save per class
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create directories
    directories = create_emotion_directories(output_dir)
    
    # Process each emotion label
    for label in range(7):  # 0 to 6
        print(f"\nProcessing emotion {label}...")
        
        # Get all rows for this label
        label_data = data[data['label'] == label]
        
        # Randomly sample rows
        sample_indices = random.sample(range(len(label_data)), min(samples_per_class, len(label_data)))
        samples = label_data.iloc[sample_indices]
        
        # Save each sample image
        for idx, (_, row) in enumerate(samples.iterrows()):
            try:
                # Convert bytes to image
                image_bytes = row['image']['bytes']
                image = Image.open(io.BytesIO(image_bytes))
                
                # Generate filename with source dataset info
                filename = f"sample_{idx + 1}_{row['source']}.png"
                filepath = os.path.join(directories[label], filename)
                
                # Save image
                image.save(filepath)
                print(f"Saved {filepath}")
                
            except Exception as e:
                print(f"Error saving image for label {label}, index {idx}: {e}")

def main():
    # Create output directory
    output_dir = "emotion_samples"
    
    # Load and prepare datasets
    print("Loading datasets...")
    raf_db = load_dataset("Mat303/raf-db-tcc", split="train")
    affectnet = load_dataset("Piro17/affectnethq", split="train")
    
    # Convert to pandas DataFrames
    raf_db_df = raf_db.to_pandas()
    affectnet_df = affectnet.to_pandas()
    
    # Add source information
    affectnet_df['source'] = 'affectnet'
    raf_db_df['source'] = 'rafdb'
    
    # Map RAF-DB labels
    emotion_map = {
        6: 0,  # Angry → 0
        3: 1,  # Disgust → 1
        2: 2,  # Fear → 2
        4: 3,  # Happy → 3
        7: 4,  # Neutral → 4
        5: 5,  # Sad → 5
        1: 6   # Surprise → 6
    }
    raf_db_df['label'] = raf_db_df['label'].apply(lambda x: int(x[0]))
    raf_db_df['label'] = raf_db_df['label'].map(emotion_map)
    
    # Merge datasets
    print("Merging datasets...")
    data = pd.concat([raf_db_df, affectnet_df], ignore_index=True)
    
    # Save sample images
    print("Saving sample images...")
    save_sample_images(data, output_dir)
    
    print("\nFinished saving sample images!")
    print(f"Images are saved in the '{output_dir}' directory")

if __name__ == "__main__":
    main()