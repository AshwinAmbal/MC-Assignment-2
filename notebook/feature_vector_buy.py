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


TRIM_DATA_SIZE_BUY = 30
GESTURE = 'buy'

def feature_vector_buy_ind(trimmed_data, column_name, isBuy=False, test=False):

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

    if TRIM_DATA_SIZE_BUY - 1 > featureVector.shape[0]:
        featureVector = np.pad(featureVector, (0, TRIM_DATA_SIZE_BUY - featureVector.shape[0] - 1), 'constant')
    featureVector = featureVector[:TRIM_DATA_SIZE_BUY - 1]
    if not test:
        if isBuy:
            featureVector = np.append(featureVector, 1)
        else:
            featureVector = np.append(featureVector, 0)
    return featureVector


def feature_vector_buy(data, isBuy=False, test=False):
    trimmed_data = trim_or_pad_data(data, TRIM_DATA_SIZE_BUY)
    featureVector = feature_vector_buy_ind(trimmed_data, 'rightWrist_x', isBuy, test=True)
    featureVector = np.append(featureVector, feature_vector_buy_ind(trimmed_data, 'rightWrist_y', isBuy, test))

    return featureVector


def modeling_buy(dirPath):
    listDir = ['buy']
    featureMatrixBuy = feature_matrix_extractor(dirPath, listDir, feature_vector_buy, pos_sample=True)
    buy_df = pd.DataFrame(featureMatrixBuy)

    # Number of negative samples per folder needed to balance the dataset with positive and negative samples
    count_neg_samples = buy_df.shape[0] / 3
    listDir = ['communicate', 'really', 'hope', 'mother', 'fun']
    featureMatrixNotBuy = feature_matrix_extractor(dirPath, listDir, feature_vector_buy, pos_sample=False,
                                                      th=count_neg_samples)
    not_buy_df = pd.DataFrame(featureMatrixNotBuy)

    final_df = pd.concat([buy_df, not_buy_df], ignore_index=True)
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
    pred_labels = clf.predict(final_df.iloc[train_size:, :])
    true_labels = labelVector[train_size:]
    print(classification_report(true_labels, pred_labels))


# TEST Function:
modeling_buy(os.path.abspath('../JSON'))
