import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from src.pca_reduction import PCAReduction
import os
from src.utils import general_normalization, universal_normalization, trim_or_pad_data,	feature_matrix_extractor
from sklearn.metrics import classification_report


TRIM_DATA_SIZE_FUN = 150


def feature_vector_fun(data, isFun):
    pass


def modeling_fun(dirPath):
    listDir = ['fun']
    featureMatrixFun = feature_matrix_extractor(dirPath, listDir, feature_vector_fun, pos_sample=True)
    fun_df = pd.DataFrame(featureMatrixFun)

    # Number of negative samples per folder needed to balance the dataset with positive and negative samples
    count_neg_samples = fun_df.shape[0] / 5
    listDir = ['communicate', 'really', 'hope', 'mother', 'buy']
    featureMatrixNotFun = feature_matrix_extractor(dirPath, listDir, feature_vector_fun, pos_sample=False,
                                                      th=count_neg_samples)
    not_fun_df = pd.DataFrame(featureMatrixNotFun)

    final_df = pd.concat([fun_df, not_fun_df], ignore_index=True)
    shuffled_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    labelVector = shuffled_df.pop(shuffled_df.shape[1]-1)
    labelVector = labelVector.astype(int).tolist()

    final_df, pca, minmax = PCAReduction(shuffled_df)

    # clf = svm.SVC(random_state=42, probability=True)
    # clf = svm.SVC(random_state=42)
    clf = LogisticRegression(random_state=42)
    # 70:30 Train-Test Split
    train_size = int(final_df.shape[0] * 70 / 100)
    clf.fit(final_df.iloc[:train_size, :], labelVector[:train_size])

    # print(clf.predict_proba(final_df.iloc[train_size:, :]))
    pred_labels = clf.predict(final_df.iloc[train_size:, :])
    true_labels = labelVector[train_size:]

    print(classification_report(true_labels, pred_labels))


# TEST Function:
modeling_fun(os.path.abspath('../JSON'))
