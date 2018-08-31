# inference
## neural network training
import pandas as pd
import numpy as np
from skimage.io import imread

from keras.models import load_model
from skimage.transform import resize
from helper_functions import (feature_engineering, clustering, find_cluster)
import pylab as plt
import os
import sys
import pickle
import seaborn as sns
from sklearn.externals import joblib
sns.set_style("white")

class Inference:

    def __init__(self, model, image_path, label2breed, invertlabel2class,
                 kmeans_hog = None, kmeans_sift = None, kmeans_daisy = None,
                 l_svm = False):
        self.image_path = image_path
        self.model = model
        self.invertlabel2class = invertlabel2class
        self.label2breed = label2breed
        self.kmeans_sift = kmeans_sift
        self.kmeans_daisy= kmeans_daisy
        self.kmeans_hog = kmeans_hog
        self.l_svm = l_svm

    def feature_engineering(self):
        X = []

        if self.l_svm:
            output = feature_engineering(self.image_path)
            bovw_hog = find_cluster(self.kmeans_hog,
                                    np.reshape(output[0],(-1,1)))
            bovw_daisy = find_cluster(self.kmeans_daisy, output[1])
            bovw_sift = find_cluster(self.kmeans_sift, output[2])
            bovw_feature = bovw_sift + bovw_daisy + bovw_hog
            X.append(bovw_feature)

        else:
            mat_img = np.array(self.image_path)
            X.append(resize(mat_img, (224,224,3)))

        return np.array(X)

    def predict(self, X, k = 5):
        if self.l_svm:
            y_pred = self.model.predict_proba(X)
        else:
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
        image_path = dir_path + "/" + sys.argv[1]
    except IndexError:
        image_temp = input("Please enter image path:")
        image_path = dir_path + "/" + image_temp
    print("Reading from ", image_path)

    try:
        assert os.path.exists(image_path) == True
    except AssertionError:
        print("Your file does not exist. Please relaunch with a correct file.")
        sys.exit(0)

    # Image or text output
    yes = {'yes','y', 'ye'}
    no = {'no','n', ''}

    choice = input("Output image? [y/N] [ENTER] ").lower()
    if choice in yes:
        l_visual = True
    elif choice in no:
        l_visual = False
    else:
        l_visual = True
        print("Please respond with 'yes' or 'no'")

    # image,miscellaneous files
    image = imread(image_path)
    invertlabel2class, label2breed = load_miscellaneous(dir_path)
    
    # Neural network
    model_nn = load_model(dir_path + "/db/nn_xception.h5")
    inference_nn = Inference(model_nn, image, label2breed, invertlabel2class)
    X_nn  = inference_nn.feature_engineering()
    breed_nn, y_pred_nn = inference_nn.predict(X_nn)
    y_pred_sort_nn = np.sort(y_pred_nn)[::-1]
    index_y_pred_sort_nn = np.argsort(y_pred_nn)[::-1]
    df_pred_nn = pd.DataFrame(list(zip(np.array(breed_nn)[index_y_pred_sort_nn],
                            np.round(y_pred_sort_nn,2))),
                            columns = ["BREED","PROBA"])

    # SVM
    model_svm = joblib.load(dir_path + '/db/svm_linear.sav')
    kmeans_hog = joblib.load(dir_path + '/db/kmeans_hog.sav')
    kmeans_sift = joblib.load(dir_path + '/db/kmeans_sift.sav')
    kmeans_daisy= joblib.load(dir_path + '/db/kmeans_daisy.sav')
    inference_svm = Inference(model_svm, image, label2breed, invertlabel2class,
                             kmeans_hog = kmeans_hog, kmeans_sift= kmeans_sift,
                             kmeans_daisy = kmeans_daisy,l_svm =True)
    X_svm  = inference_svm.feature_engineering()
    breed_svm, y_pred_svm = inference_svm.predict(X_svm)
    y_pred_sort_svm = np.sort(y_pred_svm)[::-1]
    index_y_pred_sort_svm = np.argsort(y_pred_svm)[::-1]
    df_pred_svm = pd.DataFrame(list(zip(np.array(breed_svm)[index_y_pred_sort_svm],
                            np.round(y_pred_sort_svm,2))),
                            columns = ["BREED","PROBA"])

    # API-Plot
    if l_visual:
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        ax[0].imshow(image)
        ax[0].axis('off')

        sns.barplot(x = "PROBA", y = "BREED", ax=ax[1], data=df_pred_svm)
        ax[1].set_xlabel("Probability")
        ax[1].set_ylabel("")
        ax[1].set_title("SVM (linear)", fontsize=14)

        sns.barplot(x = "PROBA", y = "BREED", ax=ax[2], data=df_pred_nn)
        ax[2].set_xlabel("Probability")
        ax[2].set_ylabel("")
        ax[2].set_title("XCEPTION", fontsize=14)

        sns.despine()
        plt.tight_layout()
        plt.show()

    else:
        title_svm = "Using engineered features (Linear SVM)"
        print(len(title_svm)*"*")
        print(title_svm)
        print(len(title_svm)*"*")
        print(df_pred_svm)

        title_nn = "Using Neural Network (XCEPTION)"
        print(len(title_nn)*"*")
        print(title_nn)
        print(len(title_nn)*"*")
        print(df_pred_nn)
