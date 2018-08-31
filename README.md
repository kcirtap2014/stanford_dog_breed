# stanford_dog_breed

This repo contains files used to train 20 classes of dog breeds. The dataset is from Stanford Dog Explorer.

Two approaches were studies: 
- classical approach: feature engineering with decriptors + classfication algortithms
- CNN approach: choices of CNN are VGG16 and Xception

## Files:
- db: folder needed to launch predict.py
- predict.py: file used for prediction, it takes an image as input and outputs breed probability 
- nn_training.py: python file used for neural network training
- helper_functions.py: annexe functions
- EDA.ipynb: exploratory ipython notebook for the topic
