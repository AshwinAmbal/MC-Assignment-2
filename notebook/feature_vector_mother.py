import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from notebook.pca_reduction import PCAReduction
import os
from notebook.utils import general_normalization, universal_normalization, trim_or_pad_data
from notebook.utils import feature_matrix_extractor, modelAndSave
from sklearn.metrics import classification_report


TRIM_DATA_SIZE_MOTHER = 120
GESTURE = 'mother'


def feature_vector_mother(data, isMother=False, test=False):
	trimmed_data = trim_or_pad_data(data, TRIM_DATA_SIZE_MOTHER)
	rY = trimmed_data['rightWrist_y']

	normRawColumn = general_normalization(rY)
	normRawColumn = universal_normalization(normRawColumn, trimmed_data, x_norm=False)

	diffNormRawData = np.diff(normRawColumn)

	zeroCrossingArray = np.array([])
	maxDiffArray = np.array([])

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
	featureVector = np.append(featureVector, diffNormRawData)
	featureVector = np.append(featureVector, zeroCrossingArray[index[0:5]])
	featureVector = np.append(featureVector, maxDiffArray[index[0:5]])
	if TRIM_DATA_SIZE_MOTHER - 1> featureVector.shape[0]:
		featureVector = np.pad(featureVector, (0, TRIM_DATA_SIZE_MOTHER - featureVector.shape[0] - 1), 'constant')
	featureVector = featureVector[:TRIM_DATA_SIZE_MOTHER-1]
	if not test:
		if isMother:
			featureVector = np.append(featureVector, 1)
		else:
			featureVector = np.append(featureVector, 0)
	return featureVector


def modeling_mother(dirPath):
	listDir = ['mother']
	featureMatrixMother = feature_matrix_extractor(dirPath, listDir, feature_vector_mother, pos_sample=True)
	mother_df = pd.DataFrame(featureMatrixMother)

	# Number of negative samples per folder needed to balance the dataset for positive and negative samples
	count_neg_samples = mother_df.shape[0] / 3
	listDir = ['communicate', 'really', 'hope', 'fun', 'buy']
	featureMatrixNotMother = feature_matrix_extractor(dirPath, listDir, feature_vector_mother, pos_sample=False,
													  th=count_neg_samples)
	not_mother_df = pd.DataFrame(featureMatrixNotMother)

	final_df = pd.concat([mother_df, not_mother_df], ignore_index=True)
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
modeling_mother(os.path.abspath('../JSON'))
