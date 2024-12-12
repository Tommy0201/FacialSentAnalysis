from datasets import load_dataset
import pandas as pd
import io
from PIL import Image
import matplotlib.pyplot as plt
# https://pure.port.ac.uk/ws/files/29163605/Relation_Aware_Facial_Expression_Recognition_PP.pdf
# https://rua.ua.es/dspace/bitstream/10045/136107/6/Mejia-Escobar_etal_2023_IEEE-Access.pdf

# RAF-DB
# https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf
raf_db = load_dataset("Mat303/raf-db-tcc", split="train")
raf_db_df = raf_db.to_pandas()
# Total samples: 15,339
# 7 classes: 1: Surprise - 2: Fear - 3: Disgust - 4: Happy - 5: Sad - 6: Angry - 7: Neutral

# AffectNet
affectnet = load_dataset("Piro17/affectnethq", split="train")
affectnet_df = affectnet.to_pandas()
# Total samples: 27,823
# 7 classes: 0: Angry - 1: Disgust - 2: Fear - 3: Happy - 4: Neutral - 5: Sad - 6: Surprise

# Modify AffectNet
# print(type(affectnet_df['label'][0]))
# affectnet_df['label'] = affectnet_df['label'].apply(lambda x: int(x))
# print(type(affectnet_df['label'][0]))
affectnet_df['source'] = 'affectnet'

# Modify RafDB
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
print(type(raf_db_df['label'][0]))
raf_db_df['source'] = 'rafdb'

# Merge datasets
data = pd.concat([raf_db_df, affectnet_df], ignore_index=True) # 43,162 samples
# print("First row:")
# print(data.head(2))  # head(1) returns the first row

# bytes1 = data.head(1)['image'].iloc[0]['bytes']
# image1 = Image.open(io.BytesIO(bytes1))
# image1.save("image1.png", "PNG")


# Print the last row
print("Last row:")
print(data.iloc[30000])  # tail(1) returns the last row
data.to_csv("training_models/data/data2.csv", index=False)

bytes2 = data.iloc[30000]['image']['bytes']
image2 = Image.open(io.BytesIO(bytes2))
image2.save("image2.png", "PNG")


# 7 classes - affectnet
# 0: Angry - 
# 1: Disgust - 
# 2: Fear - 
# 3: Happy - 
# 4: Neutral - 
# 5: Sad - 
# 6: Surprise

# affectnet (27,823)
# rafdb (15,339)


import matplotlib.pyplot as plt
import seaborn as sns

# Emotion Class Distribution for Each Source
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

for ax, source in zip(axes, data['source'].unique()):
    sns.countplot(data=data[data['source'] == source], x='label', ax=ax, palette='muted')
    ax.set_title(f'Emotion Class Distribution in {source.upper()}', fontsize=14)
    ax.set_xlabel('Emotion Classes', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"], rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# Distribution of Emotion Classes
class_counts = data['label'].value_counts()
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

plt.figure(figsize=(10, 6))
sns.barplot(x=class_names, y=class_counts.values, palette="viridis")
plt.title("Class Distribution")
plt.xlabel("Emotion Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.show()

# Sample Images with Emotion Labels
import random
emotion_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
fig, axes = plt.subplots(7, 5, figsize=(15, 20))

for emotion, ax_row in zip(range(7), axes):
    emotion_samples = data[data['label'] == emotion].sample(5, random_state=42)
    for ax, (_, row) in zip(ax_row, emotion_samples.iterrows()):
        img = Image.open(io.BytesIO(row['image']['bytes']))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Class: {emotion_names[emotion]}")  # Change title to emotion name

plt.tight_layout()
plt.show()
