import numpy as np
import pandas as pd
from skimage.color import rgb2grey
from skimage.feature import daisy, hog
from sklearn.cluster import MiniBatchKMeans
from skimage import filters
import matplotlib.pyplot as plt
from skimage import exposure
import re
import cv2

def clustering(features, n_cluster):
    """
    perform clustering using MiniBatchKMeans

    Parameters:
    -----------
    features: array_like
        array of features

    n_cluster: int
        number of clusters

    Returns:
    --------
    kmeans: scikit learn model
    """
    kmeans = MiniBatchKMeans(n_cluster, batch_size=n_cluster * 10)
    kmeans.fit(features)

    return kmeans

def find_cluster(cluster_model, features):
    """
    find to which clusters each feature belongs

    Parameters:
    -----------
    cluster_model: scikit-learn model

    features: array_like
        array of features
    """
    img_clusters = cluster_model.predict(features)
    cluster_freq_counts = pd.DataFrame(
        img_clusters, columns=['cnt'])['cnt'].value_counts()
    bovw_vector = np.zeros(cluster_model.n_clusters)

    ##feature vector of size as the total number of clusters
    for key in cluster_freq_counts.keys():
        bovw_vector[key] = cluster_freq_counts[key]

    bovw_feature = bovw_vector / np.linalg.norm(bovw_vector)

    return list(bovw_feature)


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

def feature_engineering(img, step=32, radius=32, histograms=8, orientations=8,
                        visualize=False, l_hog=True, l_daisy=True, l_sift=True):
    """
    feature engineering with HOG, DAISY and/or SIFT descriptors

    Parameters:
    -----------
    img: input image

    step: int
        daisy descriptor parameter. it defines the step between descriptors

    radius: int
        daisy descriptor parameter. it defines the radius of the descriptor

    histograms: int
        daisy descriptor parameter. number of histograms per descriptor

    orientations: int
        daisy descriptor parameter. number of orientations per descriptor. each
        orientation is 45°

    visualize: boolean
        true if want to return image

    l_hog: boolean
        true if use HOG descriptor

    l_daisy: boolean
        true if use DAISY descriptor

    l_sift: boolean
        true if use SIFT descriptor

    Return:
    -------
    feature descriptors
    """
    mat_img_filter = image_preprocessing(img)
    mat_img = np.array(img)
    output = []

    if l_hog:
        if visualize:
            fd, img_hog = hog(mat_img_filter, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=visualize,
                              feature_vector=True)
            output_hog = fd, img_hog

        else:
            fd = hog(mat_img_filter, orientations=8, pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1), visualize=visualize,
                     feature_vector=True)
            output_hog = fd

        output.append(output_hog)

    if l_daisy:
        # apply daisy feature extraction
        if visualize:
            descs, img_daisy = daisy(mat_img_filter, step=step, rings=2,
                                     histograms=histograms, radius=radius,
                                     normalization='l2',
                                     orientations=orientations,
                                     visualize=visualize)

            output_daisy = descs, img_daisy

        else:
            descs = daisy(mat_img_filter, step=step, rings=2, radius=radius,
                          histograms=histograms, normalization='l2',
                          orientations=orientations, visualize=visualize)

            descs_num = descs.shape[0] * descs.shape[1]
            daisy_descriptors = descs.reshape(descs_num, descs.shape[2])
            output_daisy = daisy_descriptors

        output.append(output_daisy)

    if l_sift:

        sift = cv2.xfeatures2d.SIFT_create()
        # convert to gray scale
        img_gray = cv2.cvtColor(mat_img, cv2.COLOR_BGR2GRAY)

        # denoise
        #median_blur_img = cv2.medianBlur(img_gray, ksize=1)

        # equalizer: contrast adjustment
        img_eq = cv2.equalizeHist(img_gray)

        kp, descs = sift.detectAndCompute(img_eq, None)
        output_sift = descs

        if visualize:
            img_sift = cv2.drawKeypoints(img_eq, kp, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            output_sift = descs, img_sift

        output.append(output_sift)

    return output

def group_sort(df, feature, by_index=False, ascending=True, lim=None):
    """
    Group by features, and sort that in descending order with respect to size

    Parameters:
    -----------
    df: pandas dataframe
        input dataframe

    feature : str
        name of feature

    ascending: boolean
        ascending or descending order

    lim: int
        Default: None

    Return:
    ------
    db_sorted: pandas dataframe
        filtered_database
    """
    db = df.groupby([feature]).size()

    if by_index:
        db_sorted = db.sort_index(ascending=ascending)

    else:
        db_sorted = db.sort_values(ascending=ascending)

    if lim is not None:
        db_sorted = db_sorted.head(lim)

    return db_sorted

def image_preprocessing(img):

    mat_img = np.array(img)

    # convert to gray scale as only the luminosity is important
    mat_img_gray = rgb2grey(mat_img)

    # denoise
    #mat_img_filter = filters.median(mat_img_gray)

    # contast adjustment
    mat_img_eq = exposure.equalize_adapthist(mat_img_gray, clip_limit=0.03)

    return mat_img_eq

def learning_curve(n_clusters, train_scores, test_scores, ax):
    """
    plot learning curve

    Parameters:
    -----------
    train_scores: array_like
        train_scores
    test_scores: array_like
        test_scores
    ax: plt axis object
        plotting axis

    Returns:
    --------
    ax: plt axis object
    """

    train_scores_mean = np.array([np.mean(v) for v in train_scores])
    train_scores_std = np.array([np.std(v) for v in train_scores])
    test_scores_mean = np.array([np.mean(v) for v in test_scores])
    test_scores_std = np.array([np.std(v) for v in test_scores])
    #cv_scores_mean = np.array([np.mean(v) for v in cv_scores])
    #cv_scores_std = np.array([np.std(v) for v in cv_scores])

    ax.fill_between(n_clusters, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    #ax.fill_between(n_clusters, cv_scores_mean - cv_scores_std,
    #                     cv_scores_mean + cv_scores_std, alpha=0.1, color="g")
    ax.fill_between(n_clusters, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="b")
    ax.plot(n_clusters, train_scores_mean, 'o-', color="r", label="Training")
    #ax.plot(n_clusters, cv_scores_mean, 'o-', color="g",
    #             label="Cross-validation")
    ax.plot(n_clusters, test_scores_mean, 'o-', color="b", label="Test")

    ax.legend(loc='best')

    ax.set_xlabel("# clusters")
    ax.set_ylabel("Score")

    return ax

def img_cropping(parsedXML, img):
    """
    image cropping using bounding boxes provided by
    """
    xmin = int(parsedXML.getElementsByTagName('xmin')[0].firstChild.nodeValue)
    ymin = int(parsedXML.getElementsByTagName('ymin')[0].firstChild.nodeValue)
    xmax = int(parsedXML.getElementsByTagName('xmax')[0].firstChild.nodeValue)
    ymax = int(parsedXML.getElementsByTagName('ymax')[0].firstChild.nodeValue)

    try:
        cropped = img[xmin:xmax, ymin:ymax]
    except ValueError:
        # take the initial picture if anomalies
        cropped = img

    return cropped

def find_dogbreed(file_str, delimiter="-|/"):
    return re.split("-|/",file_str)[1]
