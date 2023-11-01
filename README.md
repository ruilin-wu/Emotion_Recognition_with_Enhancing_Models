# Emotion Recognition With Enhancing Models

**Introduction**

This project builds on the work of this project https://github.com/marcogdepinto/emotion-classification-from-audio-files and proposes a more accurate deep learning classifier capable of predicting human speech encoded in audio files the person’s emotions. This classifier uses 2 different data sets RAVDESS (song subset) and TESS, uses a more advanced CNN model, and extracts MFCC features from the file for model training. The overall F1 score on 8 categories (neutral, calm, happy, sad, angry, fearful, disgusted, and surprised) is 96%.

**Feature set information**

For this task, the dataset is built using 5252 samples from:

- the [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset](https://zenodo.org/record/1188976#.XsAXemgzaUk) 
- the [Toronto emotional speech set (TESS) dataset](https://tspace.library.utoronto.ca/handle/1807/24487) 


- For **RAVDESS** database, the dataset incorporated 1440 speech files and 1012 song files. This collection features recordings by 24 professional actors, comprising 12 females and 12 males. They vocalized two statements that matched in terms of lexicon, delivered in a neutral North American accent. The speech recordings encompassed various emotions such as calm, happy, sad, angry, fearful, surprise, and disgust. Meanwhile, the song recordings covered emotions like calm, happy, sad, angry, and fearful. Each of these files underwent a rating process, evaluated 10 times based on emotional validity, intensity, and authenticity. The ratings were given by 247 individuals, typical of untrained adult participants from North America. An additional group of 72 participants contributed data for test-retest reliability. 

- For **TESS** database, the dataset integrated 2800 files. Two actresses, aged 26 and 64, vocalized 200 target words within the carrier phrase "Say the word ". These recordings captured seven distinct emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral, resulting in a total of 2800 stimuli. Both actresses, hailing from Toronto, are native English speakers with university education and musical training backgrounds. Audiometric tests confirmed that both had normal hearing thresholds.

The objective of the model is to classify the recordings into the following emotion categories: 0 for neutral, 1 for calm, 2 for happy, 3 for sad, 4 for angry, 5 for fearful, 6 for disgust, and 7 for surprised. It's important to note that the dataset is imbalanced. The TESS database lacks the calm category, leading to fewer samples for that emotion. 

**Metrics**

*Model summary*

![Link to model](https://github.com/ruilin-wu/Emotion_Recognition_with_Enhancing_Models/blob/main/media/model_project.png) 

*Loss and accuracy plots*

![Link to loss](https://github.com/ruilin-wu/Emotion_Recognition_with_Enhancing_Models/blob/main/media/loss.png) 

![Link to accuracy](https://github.com/ruilin-wu/Emotion_Recognition_with_Enhancing_Models/blob/main/media/accuracy.png)

*Classification report*

![Link do classification report](https://github.com/ruilin-wu/Emotion_Recognition_with_Enhancing_Models/blob/main/media/classification_report1.png)

*Confusion matrix*

![Link do classification report](https://github.com/ruilin-wu/Emotion_Recognition_with_Enhancing_Models/blob/main/media/confusion_matrix.png)

**How to use the code inside this repository**

1) To install this repository, please use this command:
 ```git clone https://github.com/ruilin-wu/Emotion_Recognition_with_Enhancing_Models.git ```

2) Use ```pip install -r requirements.txt``` to install the required libraries.

3) Use command ```python tess_pipeline.py``` to rename the files in the TESS database and send them to ```features\Actor_25``` and ```features\Actor_26```. Please do not run this script beacuse these files already exist in the Actor_25 and Actor_26 folders of this project.

4) Use command ```python create_features.py``` to extract MFCCs from each file and save them as .joblib files. In addition to feature MFCC, there are three other feature extraction methods in ```create_features.py```: chroma, contrast and the combination of MFCC, chroma, and contrast.

5) Use command ```python neural_network.py``` to load the .joblib file and use machine learning architecture to train H5 model. And 
 ```neural_network.py``` proposes four models, and this project found that the best model is the Improved CNN model.

6) Use command ```python live_predictions.py``` to use the pre-trained H5 model to perform emotion recognition on real-time audio files.
    
*Prediction results*

![Link to model](https://github.com/ruilin-wu/Emotion_Recognition_with_Enhancing_Models/blob/main/media/prediction_result.png) 


7) Use command ```python plot_model.py``` to load the trained H5 model, draw and save the structure diagram of the model .

*Model summary*

![Link to model](https://github.com/ruilin-wu/Emotion_Recognition_with_Enhancing_Models/blob/main/media/model_project.png) 


**SUMMARY Of The PROJECT**

For more detailed analysis, please refer to the project report that comes with the project, but the best model trained in this project uses the following parameters: (1) In RAVDESS, we used song subset; (2) Among various features, this project chose MFCC Instead of other features; (3) In ```neural_network.py```, this project provides four different models. Finally, it was found that the improved CNN model has the best effect. Readers can test the performance of other models at will.




**APPENDIX 1: The RAVDESS dataset**

Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.

*Filename identifiers*

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).

Vocal channel (01 = speech, 02 = song).

Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).

Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.

Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).

Repetition (01 = 1st repetition, 02 = 2nd repetition).

Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
```
@article{RAVDESS_2018,
author = {Livingstone SR, Russo FA},
journal = {PLoS ONE},
title = "{The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English}",
year = {2018},
volume = {13},
issue = {5},
pages = {e0196391},
doi = {10.1371/journal.pone.0196391},
url = {https://doi.org/10.1371/journal.pone.0196391}
}
```

**APPENDIX 2: The TESS dataset**

Pichora-Fuller, M. Kathleen; Dupuis, Kate, 2020, "Toronto emotional speech set (TESS)", https://doi.org/10.5683/SP2/E8H2MF, Scholars Portal Dataverse, V1

```
@data{SP2/E8H2MF_2020,
author = {Pichora-Fuller, M. Kathleen and Dupuis, Kate},
publisher = {Scholars Portal Dataverse},
title = "{Toronto emotional speech set (TESS)}",
year = {2020},
version = {DRAFT VERSION},
doi = {10.5683/SP2/E8H2MF},
url = {https://doi.org/10.5683/SP2/E8H2MF}
}
```

**APPENDIX 3: M. G. de Pinto's Work**

M. G. de Pinto, “Audio Emotion Classification from Multiple Datasets,” GitHub, Oct. 22, 2023. https://github.com/marcogdepinto/emotion-classification-from-audio-files (accessed Oct. 28, 2023).
```
@INPROCEEDINGS{9122698,
author={M. G. {de Pinto} and M. {Polignano} and P. {Lops} and G. {Semeraro}},
booktitle={2020 IEEE Conference on Evolving and Adaptive Intelligent Systems (EAIS)},
title={Emotions Understanding Model from Spoken Language using Deep Neural Networks and Mel-Frequency Cepstral Coefficients},
year={2020},
volume={},
number={},
pages={1-5},}
```
