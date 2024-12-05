from datasets import load_dataset
import pandas as pd
# https://pure.port.ac.uk/ws/files/29163605/Relation_Aware_Facial_Expression_Recognition_PP.pdf
# https://rua.ua.es/dspace/bitstream/10045/136107/6/Mejia-Escobar_etal_2023_IEEE-Access.pdf

# RAF-DB
# https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf
raf_db = load_dataset("Mat303/raf-db-tcc", split="train")
raf_db_df = raf_db.to_pandas()
# Total samples: 15,339
# 7 classes: 1: Surprise - 2: Fear - 3: Disgust - 4: Happy - 5: Sad - 6: Angry - 7: Neutral

# AffectNet
affectnethq = load_dataset("Piro17/affectnethq", split="train")
affectnethq_df = affectnethq.to_pandas()
# Total samples: 27,823
# 7 classes: 0: Angry - 1: Disgust - 2: Fear - 3: Happy - 4: Neutral - 5: Sad - 6: Surprise

# Modify RAF-DB
raf_db_df['label'] = raf_db_df['label'].apply(lambda x: int(x[0]))
raf_db_df['source'] = 'raf-db-tcc'
raf_db_df.to_csv("rafdb.csv", index=False)

# Modify AffectNet:
emotion_map = {
    0: 6,  # Angry → 6
    1: 3,  # Disgust → 3
    2: 2,  # Fear → 2
    3: 4,  # Happy → 4
    4: 7,  # Neutral → 7
    5: 5,  # Sad → 5
    6: 1   # Surprise → 1
}
affectnethq_df['label'] = affectnethq_df['label'].map(emotion_map)
affectnethq_df['source'] = 'affectnethq'
affectnethq_df.to_csv("affectnet.csv", index=False)

# Merge datasets
data = pd.concat([raf_db_df, affectnethq_df], ignore_index=True) # 43,162 samples
data.to_csv("data.csv", index=False)
