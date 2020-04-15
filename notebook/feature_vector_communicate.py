import os
import pandas as pd
import numpy as np

import tsfresh.feature_extraction.feature_calculators as fc

from scipy.fftpack import fft
from notebook.pca_reduction import PCAReduction
from notebook.utils import general_normalization, universal_normalization, trim_or_pad_data,	feature_matrix_extractor
from notebook.utils import modelAndSave
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



TRIM_DATA_SIZE_COMMUNICATE = 80
GESTURE = 'communicate'

def feature_vector_communicate_ind(trimmed_data, column_name, iscommunicate=False, test=False):

    r = trimmed_data[column_name]

    normRawColumn = general_normalization(r)
    normRawColumn = universal_normalization(normRawColumn, trimmed_data, x_norm=True)

    diffNormRawData = np.diff(normRawColumn)

    zeroCrossingArray = np.array([])
    maxDiffArray = np.array([])

    # Fast Fourier Transform
    fftArray = np.array([])
    fftVal = []
    fft_coefficients = fft(diffNormRawData, n=6)[1:]
    fft_coefficients_real = [value.real for value in fft_coefficients]
    fftVal += fft_coefficients_real
    fftArray = np.append(fftArray, fftVal)

    # Windowed Mean for each second of the video
    windowedVal = np.array([])
    for i in range(0,diffNormRawData.shape[0],30):
        windowedVal = np.append(windowedVal, fc.mean(diffNormRawData[i:i+30]))

    # Other features
    if diffNormRawData[0] > 0:
        initSign = 1
    else:
        initSign = 0

    windowSize = 5

    for x in range(1, len(diffNormRawData)):

        if diffNormRawData[x] > 0:
            newSign = 1
        else:
            newSign = 0

        if initSign != newSign:
            zeroCrossingArray = np.append(zeroCrossingArray, x)
            initSign = newSign
            maxIndex = np.minimum(len(diffNormRawData), x + windowSize)
            minIndex = np.maximum(0, x - windowSize)

            maxVal = np.amax(diffNormRawData[minIndex:maxIndex])
            minVal = np.amin(diffNormRawData[minIndex:maxIndex])

            maxDiffArray = np.append(maxDiffArray, (maxVal - minVal))

    index = np.argsort(-maxDiffArray)

    featureVector = np.array([])
    featureVector = np.append(featureVector, fftArray)
    featureVector = np.append(featureVector, windowedVal)
    featureVector = np.append(featureVector, zeroCrossingArray[index[0:5]])
    featureVector = np.append(featureVector, maxDiffArray[index[0:5]])
    # featureVector = np.append(featureVector, diffNormRawData)

    if TRIM_DATA_SIZE_COMMUNICATE - 1 > featureVector.shape[0]:
        featureVector = np.pad(featureVector, (0, TRIM_DATA_SIZE_COMMUNICATE - featureVector.shape[0] - 1), 'constant')
    featureVector = featureVector[:TRIM_DATA_SIZE_COMMUNICATE - 1]
    if not test:
        if iscommunicate:
            featureVector = np.append(featureVector, 1)
        else:
            featureVector = np.append(featureVector, 0)
    return featureVector


def feature_vector_communicate(data, iscommunicate=False, test=False):
    trimmed_data = trim_or_pad_data(data, TRIM_DATA_SIZE_COMMUNICATE)

    featureVector = feature_vector_communicate_ind(trimmed_data, 'rightWrist_x', iscommunicate, test=True)
    featureVector = np.append(featureVector, feature_vector_communicate_ind(trimmed_data, 'rightWrist_y', iscommunicate, test=True))
    featureVector = np.append(featureVector, feature_vector_communicate_ind(trimmed_data, 'leftWrist_y', iscommunicate, test=True))
    featureVector = np.append(featureVector, feature_vector_communicate_ind(trimmed_data, 'leftWrist_y', iscommunicate, test))

    return featureVector


def modeling_communicate(dirPath):
    listDir = ['communicate']
    featureMatrixcommunicate = feature_matrix_extractor(dirPath, listDir, feature_vector_communicate, pos_sample=True)
    communicate_df = pd.DataFrame(featureMatrixcommunicate)

    # Number of negative samples per folder needed to balance the dataset with positive and negative samples
    count_neg_samples = communicate_df.shape[0] / 5
    listDir = ['fun', 'really', 'hope', 'mother', 'buy']
    featureMatrixNotCommunicate = feature_matrix_extractor(dirPath, listDir, feature_vector_communicate, pos_sample=False,
                                                      th=count_neg_samples)
    not_communicate_df = pd.DataFrame(featureMatrixNotCommunicate)

    final_df = pd.concat([communicate_df, not_communicate_df], ignore_index=True)
    shuffled_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    labelVector = shuffled_df.pop(shuffled_df.shape[1]-1)
    labelVector = labelVector.astype(int).tolist()

    final_df, pca, minmax = PCAReduction(shuffled_df)

    modelAndSave(final_df, labelVector, GESTURE, pca, minmax)
    #
    # clf = svm.SVC(random_state=42, probability=True)
    # clf = svm.SVC(random_state=42)
    # clf = LogisticRegression(random_state=42)
    clf = MLPClassifier(max_iter=5000, random_state=42)
    # clf = GaussianNB()
    # clf.fit(final_df, labelVector)
    # 70:30 Train-Test Split
    train_size = int(final_df.shape[0] * 70 / 100)
    clf.fit(final_df.iloc[:train_size, :], labelVector[:train_size])
    pred_labels = clf.predict(final_df.iloc[train_size:, :])
    true_labels = labelVector[train_size:]
    print(classification_report(true_labels, pred_labels))


# TEST Function:
# modeling_communicate(os.path.abspath('../JSON'))
