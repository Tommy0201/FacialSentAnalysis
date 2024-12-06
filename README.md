**training_models:**
Containing all the directories and files that were used for training
    **data folder:**
    Containing the data files in csv format, including: _affectnet.csv; data.csv; rafdb.csv_
    And contain the _demo.py_ file for data merging and labeling, the _tokenize-data.py_

    **emo-model folder**
    Containing our customized model for facial recognition in model.py
    As well as the training results of it. 

    **eNet-model folders**
    Containing the efficientNet pretrained model which is then applied Attention Mechanism to the last layer and finetuned it.
    Training results are included

    **ResNet50-model folders**
    Containing the ResNet50 pretrained model which is then applied Attention Mechanism to the last layer and finetuned it.
    Training results are included

**using_models:**
If you wish to use live video facial emotion detection, please run main.py

**Files that is not in the folder:**
Should include: visualize.py ; _preprocessed_data_2.pkl_ (resulted from running _tokenized_data.py_ in data folder)




