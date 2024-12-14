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

print("Unique labels in the combined dataset:")
print(data['label'].unique())
print("\nValue counts for labels:")
print(data['label'].value_counts().sort_index())

print("\nUnique sources:")
print(data['source'].unique())
print("\nValue counts for sources:")
print(data['source'].value_counts())

# data.to_csv("training_models/data/data2.csv", index=False)

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
