**training_models:**
Containing all the directories and files that were used for training
    **data folder:**
    Containing the data files in csv format, including: _affectnet.csv; data.csv; rafdb.csv_
    And contain the _demo.py_ file for data merging and labeling, the _tokenize-data.py_

    All customized model: use CNN + MultiHead Attention layer
    All pretrained model: add in Spatial Attention layer
    
    **stage1 folder:**
        + emo-model (customized)
        + eNet-model (pretrained)
        Trained on 48x48 imaegs, label 1-7


    **stage2 folder:**
        + emo-model-resize (customized)
        + eNetB4 (pretrained)
        + resNet50 (pretrained)
        Trained on 224x224 images, Label 1-7

    **stage3 folder:**
        + emo-model-2 (customized)
        + eNetB0-2 (pretrained)
        + eNetB4
        Trained on 48x48 image, Label 0-6


    **ResNet50-model folders**
    Containing the ResNet50 pretrained model which is then applied Attention Mechanism to the last layer and finetuned it.
    Training results are included

**using_models:**
If you wish to use live video facial emotion detection, please 
cd using_model
run main.py

**Files that is not in the folder:**
Should include: visualize.py ; _preprocessed_data_2.pkl_ (resulted from running _tokenized_data.py_ in data folder)




