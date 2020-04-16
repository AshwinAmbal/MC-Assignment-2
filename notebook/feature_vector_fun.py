import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from notebook.pca_reduction import PCAReduction
import os
from notebook.utils import general_normalization, universal_normalization, trim_or_pad_data,	feature_matrix_extractor
from notebook.utils import modelAndSave
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from scipy import integrate
import tsfresh.feature_extraction.feature_calculators as fc

TRIM_DATA_SIZE_FUN = 50
GESTURE = 'fun'


def feature_vector_fun(data, isFun=False, test=False):
    trimmed_data = trim_or_pad_data(data, TRIM_DATA_SIZE_FUN)
    rX = trimmed_data['rightWrist_x']

    normRawColumn = universal_normalization(rX, trimmed_data, x_norm=True)
    normRawColumn = general_normalization(normRawColumn)

    # Area under curve
    auc = np.array([])
    auc = np.append(auc, abs(integrate.simps(normRawColumn, dx=5)))

    # Absolute Sum of Consecutive Differences
    scd = fc.absolute_sum_of_changes(normRawColumn)

    # Entropy
    entropy = fc.approximate_entropy(normRawColumn, 2, 3)

    # AutoCorrelation
    ac = fc.autocorrelation(normRawColumn, lag=5)

    # Count Above Mean
    cam = fc.count_above_mean(normRawColumn)

    # Count Below Mean
    cbm = fc.count_below_mean(normRawColumn)

    featureVector = np.array([])
    featureVector = np.append(featureVector, auc)
    featureVector = np.append(featureVector, scd)
    featureVector = np.append(featureVector, entropy)
    featureVector = np.append(featureVector, ac)
    featureVector = np.append(featureVector, cam)
    featureVector = np.append(featureVector, cbm)
    if TRIM_DATA_SIZE_FUN - 1 > featureVector.shape[0]:
        featureVector = np.pad(featureVector, (0, TRIM_DATA_SIZE_FUN - featureVector.shape[0] - 1), 'constant')
    featureVector = featureVector[:TRIM_DATA_SIZE_FUN - 1]
    if not test:
        if isFun:
            featureVector = np.append(featureVector, 1)
        else:
            featureVector = np.append(featureVector, 0)
    return featureVector


def modeling_fun(dirPath):
    listDir = ['fun']
    featureMatrixFun = feature_matrix_extractor(dirPath, listDir, feature_vector_fun, pos_sample=True)
    fun_df = pd.DataFrame(featureMatrixFun)

    # Number of negative samples per folder needed to balance the dataset with positive and negative samples
    count_neg_samples = fun_df.shape[0] / 6
    listDir = ['communicate', 'really', 'hope', 'mother', 'buy']
    featureMatrixNotFun = feature_matrix_extractor(dirPath, listDir, feature_vector_fun, pos_sample=False,
                                                      th=count_neg_samples)
    not_fun_df = pd.DataFrame(featureMatrixNotFun)

    final_df = pd.concat([fun_df, not_fun_df], ignore_index=True)
    shuffled_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    labelVector = shuffled_df.pop(shuffled_df.shape[1]-1)
    labelVector = labelVector.astype(int).tolist()

    final_df, pca, minmax = PCAReduction(shuffled_df)

    modelAndSave(final_df, labelVector, GESTURE, pca, minmax)

    # clf = svm.SVC(random_state=42, probability=True)
    # clf = svm.SVC(random_state=42)
    clf = LogisticRegression(random_state=42)
    # clf = MLPClassifier(max_iter=5000, random_state=42)
    # clf = GaussianNB()
    # 70:30 Train-Test Split
    train_size = int(final_df.shape[0] * 70 / 100)
    clf.fit(final_df.iloc[:train_size, :], labelVector[:train_size])

    # print(clf.predict_proba(final_df.iloc[train_size:, :]))
    pred_labels = clf.predict(final_df.iloc[train_size:, :])
    true_labels = labelVector[train_size:]
    print(classification_report(true_labels, pred_labels))


# TEST Function:
# modeling_fun(os.path.abspath('../JSON'))
