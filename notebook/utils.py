import pandas as pd
import numpy as np
import ast
import os
import json
import pickle
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

INDEX_TO_GESTURE = {'A': 'buy', 'B': 'communicate', 'C': 'fun', 'D': 'hope', 'E': 'mother', 'F': 'really'}
GESTURE_TO_INDEX = {'buy': 'A', 'communicate': 'B', 'fun': 'C', 'hope': 'D', 'mother': 'E', 'really': 'F'}
MODEL_TO_INDEX = {'LogisticRegression': 1, 'RandomForestClassifier': 2, 'MLPClassifier': 3, 'DecisionTreeClassifier': 4}


def trim_or_pad_data(data, TRIM_DATA_SIZE):
    data = data.iloc[:TRIM_DATA_SIZE]
    if data.shape[0] < TRIM_DATA_SIZE:
        df = pd.DataFrame(np.zeros((TRIM_DATA_SIZE - data.shape[0], data.shape[1])))
        data.append(df, ignore_index=True)
    return data


def general_normalization(column):
    normRawData = (column - np.mean(column)) / (np.max(column - np.mean(column)) - np.min(column - np.mean(column)))
    return normRawData


def universal_normalization(column, data, x_norm):
    if x_norm:
        nose_x = data['nose_x']
        leftEye_x = data['leftEye_x']
        rightEye_x = data['rightEye_x']
        normRawData =  (column - nose_x) / (rightEye_x-leftEye_x)
    else:
        nose_y = data['nose_y']
        shoulder_y = data['rightShoulder_y']
        normRawData = (column - nose_y) / (nose_y - shoulder_y)
    return normRawData


def jsonToCSV(data):
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    # data = ast.literal_eval(data)
    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        row = []
        row.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            row.append(obj['score'])
            row.append(obj['position']['x'])
            row.append(obj['position']['y'])
        csv_data[i] = np.array(row)
    df = pd.DataFrame(csv_data, columns=columns)
    return df


def feature_matrix_extractor(dirPath, listDir, extractor_method, pos_sample, th=-1):
    featureMatrix = np.array([])
    for dir in listDir:
        files = os.listdir(os.path.join(dirPath, dir))
        for i, file in enumerate(files):
            if file == '.DS_Store':
                continue
            if th != -1 and i >= th:
                break
            rawDataDict = []
            for row in open(os.path.join(dirPath, dir, file)):
                rawDataDict.extend(json.loads(row))
            rawData = jsonToCSV(rawDataDict)
            # rawData = pd.read_csv(os.path.join(dirPath, dir, file), sep=',')
            featureVector = extractor_method(rawData, pos_sample)
            if featureMatrix.shape[0] == 0:
                featureMatrix = np.array([featureVector])
            else:
                featureMatrix = np.concatenate((featureMatrix, [featureVector]), axis=0)
    return featureMatrix


def modelAndSave(final_df, labelVector, gesture, pca, minmax):
    # clf = svm.SVC(random_state=42)
    # clf = svm.SVC(random_state=42, probability=True)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(final_df, labelVector)
    writePickle(clf, gesture, pca, minmax)

    clf = LogisticRegression(random_state=42)
    clf.fit(final_df, labelVector)
    writePickle(clf, gesture, pca, minmax)

    clf = MLPClassifier(max_iter=5000, random_state=42)
    clf.fit(final_df, labelVector)
    writePickle(clf, gesture, pca, minmax)

    clf = DecisionTreeClassifier()
    clf.fit(final_df, labelVector)
    writePickle(clf, gesture, pca, minmax)


def writePickle(clf, gesture, pca, minmax):
    pickle.dump(clf, open(os.path.abspath('../models/model_{}/{}/model.pkl'.format(MODEL_TO_INDEX[clf.__class__.__name__],
                                                                             GESTURE_TO_INDEX[gesture])), 'wb'))
    pickle.dump(pca, open(os.path.abspath('../models/model_{}/{}/pca.pkl'.format(MODEL_TO_INDEX[clf.__class__.__name__],
                                                                       GESTURE_TO_INDEX[gesture])), 'wb'))
    pickle.dump(minmax, open(os.path.abspath('../models/model_{}/{}/minmax.pkl'.format(MODEL_TO_INDEX[clf.__class__.__name__],
                                                                        GESTURE_TO_INDEX[gesture])), 'wb'))
