## neural network training
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
from scipy.io import loadmat
from keras.utils import to_categorical
from skimage.transform import resize
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from numpy.random import seed, shuffle
from keras.applications.vgg16 import VGG16
from sklearn.metrics import accuracy_score
import os
import re


class NeuralNetwork:

    def __init__(self, model, label2class, invertlabel2class, label2breed):
        self.model = model
        self.label2class = label2class
        self.invertlabel1class = invertlabel2class
        self.label2breed = label2breed

    def dataset_preparation(self, df, index_samples):
        X = []

        for i, key in enumerate(index_samples):
            img = Image.open("./images/" + df.loc[key].PATH)
            mat_img = np.array(img)
            X.append(resize(mat_img, (224,224,3)))

        y_temp = df.loc[index_samples].LABELS.replace(self.label2class)
        y = to_categorical(y_temp, num_classes=len(self.label2class))

        return np.array(X), np.array(y)

    def fit(self, X, y, batch_size = 32, epochs = 15, validation_data=None):
        self.model.compile(loss="categorical_crossentropy",
                           optimizer= Adam(lr=.0001),
                           metrics=['accuracy'])

        if validation_data is not None:
            model_info = self.model.fit(X, y, epochs=epochs,
                                    batch_size=batch_size, verbose=1,
                                    validation_data=validation_data)
        else:
            model_info = self.model.fit(X, y, epochs=epochs,
                                    batch_size=batch_size, verbose=1)

        return model_info

    def predict(self, X):
        y_pred = self.model.predict(X)

        return y_pred

    def score(self, y_pred, y_test):
        # one hot encoding
        y_pred_one = to_categorical([np.argmax(i) for i in y_pred],
                                    num_classes=len(self.label2class))
        score = accuracy_score(y_test, y_pred_one)

        return score

    def save(self, filename):
        dir_path = os.getcwd()
        self.model.save(dir_path +'/'+ filename +'.h5')

def find_dogbreed(file_str, delimiter="-|/"):
    return re.split("-|/",file_str)[1]

def top_k_accuracy(y_pred_proba, y_true, classes, k=5):
    """
    evaluate top k accuracy score

    Parameters:
    -----------
    y_pred_proba: array_like
        probability of y predictions

    y_true: array_like
        y true values

    classes: array_like
        map index to class number

    k: int
        number of the most preponderant predictions. Default value: 5
    """
    # decision function calculates the distance from the hyperplane, the
    # further the better, inversion is needed
    y_pred_proba_sorted = [classes[np.argsort(p)[::-1][:k]] for p in y_pred_proba]
    score = 0.

    for index, y in enumerate(y_true):
        if y in y_pred_proba_sorted[index]:
            score += 1.

    score /= np.shape(y_pred_proba_sorted)[0]

    return score

if __name__ == '__main__':
    dir_path = os.getcwd()

    # load train/test list
    train_list = loadmat(dir_path  + '/lists/train_list.mat')
    test_list = loadmat(dir_path + '/lists/test_list.mat')
    file_list_train = [entry.item() for entry in train_list["file_list"].flatten()]
    label_list_train = [entry.item() for entry in train_list["labels"]]
    df_train = pd.DataFrame(list(zip(label_list_train,
                                     file_list_train)), columns = ["LABELS","PATH"])
    df_train["BREED"] = df_train.PATH.apply(find_dogbreed)
    df_y_train = pd.get_dummies(df_train.LABELS)

    file_list_test = [entry.item() for entry in test_list["file_list"].flatten()]
    label_list_test = [entry.item() for entry in test_list["labels"]]
    df_test = pd.DataFrame(list(zip(label_list_test, file_list_test)), columns = ["LABELS","PATH"])
    df_test["BREED"] = df_test.PATH.apply(find_dogbreed)
    df_y_test = pd.get_dummies(df_test.LABELS)

    # train/test prep
    # train set
    seed(12)
    size_train = len(df_train)
    n_class_temp = np.arange(120)
    shuffle(n_class_temp)
    n_class = n_class_temp[:20]
    n_class_init = 120
    init_n_train = size_train/n_class_init # don't change
    index_train_samples = []
    index_cv_samples = []
    cv_index = 80

    # train_set_indics
    for i in n_class:
        index_train_temp = np.arange((i-1) * init_n_train, i * init_n_train , dtype=int)
        np.random.seed(2)
        np.random.shuffle(index_train_temp)
        index_train_samples.extend(index_train_temp[:cv_index])
        index_cv_samples.extend(index_train_temp[cv_index:])

    # test set indices
    n_test_overall = len(index_train_samples)
    index_test_temp = list(df_test[df_test.LABELS.isin(n_class)].index)
    np.random.shuffle(index_test_temp)
    index_test_samples = index_test_temp[:n_test_overall]

    df_train_resampled = df_train.loc[index_train_samples]
    df_test_resampled = df_test.loc[index_test_samples]

    # prepare hash table
    unique_y = df_train_resampled.LABELS.unique()
    label2class = dict()
    label2breed = dict()
    invertlabel2class = dict()

    for i, j in enumerate(unique_y):
        label2class[j] = i
        invertlabel2class[i] = j

    df_label_breed = df_train_resampled[["LABELS","BREED"]].drop_duplicates()

    for k,v in df_label_breed.iterrows():
        label2breed[v.LABELS] = v.BREED
    # load model
    modelVGG16 = VGG16(weights='imagenet', include_top=False,
                       input_shape = (224,224,3))
    # train only the last layer
    for layer in modelVGG16.layers:
        layer.trainable = False

    # adapt output to our case
    x = modelVGG16.output
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    # let's add two fully-connected layer
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    pred = Dense(20, activation="softmax")(x)
    dogbreed_model = Model(inputs=modelVGG16.input, outputs=[pred])

    model = NeuralNetwork(dogbreed_model, label2class,
                          invertlabel2class, label2breed)

    print(model.model.summary())

    print("Training data preparation")
    X_train, y_train = model.dataset_preparation(df_train, index_train_samples)

    print("CV data preparation")
    X_cv, y_cv= model.dataset_preparation(df_train, index_cv_samples)

    print("Test data preparation")
    X_test, y_test = model.dataset_preparation(df_test, index_test_samples)
    print("Fitting data...")
    model_info = model.fit(X_train, y_train, validation_data =(X_cv,y_cv))

    print("Predicting data...")
    y_pred = model.predict(X_test)

    print("Scoring model...")
    score = model.score(y_pred, y_test)

    print("Accuracy score: ", score)

    print("Saved model to ", dir_path +"/nn_model_2.h5")
    model.save("nn_model_2")
