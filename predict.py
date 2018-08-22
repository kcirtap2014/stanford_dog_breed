# inference
## neural network training
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model
from skimage.transform import resize
import pylab as plt
import os
import sys
import pickle
import seaborn as sns
sns.set_style("white")

class Inference:

    def __init__(self, model, image_path, label2breed, invertlabel2class):
        self.image_path = image_path
        self.model = model
        self.invertlabel2class = invertlabel2class
        self.label2breed = label2breed

    def feature_engineering(self):
        X = []
        mat_img = np.array(self.image_path)
        X.append(resize(mat_img, (224,224,3)))

        return np.array(X)

    def predict(self, X, k = 5):
        y_pred = self.model.predict(X)
        y_pred_index = np.argsort(y_pred.flatten())[::-1][:k]
        y_pred_sorted = np.sort(y_pred.flatten())[::-1][:k]
        breed = self.convert2breed(y_pred_index)

        return breed, y_pred_sorted

    def convert2breed(self, y_pred):
        breed = []

        for el in y_pred:
            breed.append(self.label2breed[self.invertlabel2class[el]])

        return breed

def load_miscellaneous(dir_path):

    invertlabel2class = pickle.load(open(dir_path +'/db/invertlabel2class.p', "rb" ))
    label2breed = pickle.load(open(dir_path + '/db/label2breed.p', "rb"))

    return invertlabel2class, label2breed

if __name__=="__main__":

    # some flags
    dir_path = os.getcwd()
    image_path = None

    # input processing
    try:
        image_path = dir_path + "/db/" + sys.argv[1]
    except IndexError:
        image_temp = input("Please enter image path:")
        image_path = dir_path + "/" + image_temp
        print("Reading from ", image_path)
        try:
            assert os.path.exists(image_path) == True
        except AssertionError:
            print("Your file does not exist. Please relaunch with a correct file.")
            sys.exit(1)

    # image, models
    image = Image.open(image_path)
    invertlabel2class, label2breed = load_miscellaneous(dir_path)
    model = load_model(dir_path + "/db/nn_model_base.h5")

    # Prediction
    inference = Inference(model, image, label2breed, invertlabel2class)
    X  = inference.feature_engineering()
    breed, y_pred = inference.predict(X)
    y_pred_sort = np.sort(y_pred)[::-1]
    index_y_pred_sort = np.argsort(y_pred)[::-1]
    df_pred = pd.DataFrame(list(zip(np.array(breed)[index_y_pred_sort], y_pred_sort)),
                            columns = ["breed","proba"])

    # API-Plot
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].imshow(image)
    ax[0].axis('off')
    y_pos = np.arange(len(y_pred_sort))
    sns.barplot(x = "proba", y = "breed", ax=ax[1], data=df_pred)
    ax[1].set_xlabel("Probability")
    ax[1].set_ylabel("")
    sns.despine()
    plt.tight_layout()
    plt.show()
