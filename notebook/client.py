import requests
import json
import os
from sklearn.metrics import accuracy_score


def get_data():
    gestures = ['fun', 'communicate', 'buy', 'hope', 'mother', 'really']
    overall_pred = [[]for _ in range(4)]
    overall_true = []
    for gesture in gestures:
        true_labels = []
        pred_mod_1 = []
        pred_mod_2 = []
        pred_mod_3 = []
        pred_mod_4 = []
        list_of_files = os.listdir(os.path.abspath('../JSON_Test/{}'.format(gesture)))
        dirList = [[gesture, file] for file in list_of_files]
        for row in dirList:
            if row[1] == '.DS_Store':
                continue
            with open(os.path.abspath('../JSON_Test/{}/{}'.format(row[0], row[1]))) as fp:
                jsonFile = json.load(fp)
            headers = {'Content-type': 'application/json'}
            # print(requests.post('http://127.0.0.1:9696/', data=json.dumps(jsonFile), headers=headers).json(), '\tAnswer = {}'.format(row[0]))
            result = requests.post('http://127.0.0.1:9696/', data=json.dumps(jsonFile), headers=headers).json()
            true_labels.append(row[0])
            pred_mod_1.append(result['1'])
            pred_mod_2.append(result['2'])
            pred_mod_3.append(result['3'])
            pred_mod_4.append(result['4'])
        overall_true.extend(true_labels)
        overall_pred[0].extend((pred_mod_1))
        overall_pred[1].extend((pred_mod_2))
        overall_pred[2].extend((pred_mod_3))
        overall_pred[3].extend((pred_mod_4))
        print("\n\nCLASSIFICATION RESULTS FOR GESTURE: {}".format(gesture))
        score1 = accuracy_score(true_labels, pred_mod_1)
        score2 = accuracy_score(true_labels, pred_mod_2)
        score3 = accuracy_score(true_labels, pred_mod_3)
        score4 = accuracy_score(true_labels, pred_mod_4)
        print("\nMODEL 1: ", score1)
        print("MODEL 2: ", score2)
        print("MODEL 3: ", score3)
        print("MODEL 4: ", score4)
        print("\nAVG ACCURACY: ", (score1 + score2 + score3 + score4) / 4)
        print("MAX ACCURACY: ", max(score1, score2, score3, score4))

    print('\n\nOVERALL ACCURACY: ')
    score1 = accuracy_score(overall_true, overall_pred[0])
    score2 = accuracy_score(overall_true, overall_pred[1])
    score3 = accuracy_score(overall_true, overall_pred[2])
    score4 = accuracy_score(overall_true, overall_pred[3])
    print("\nMODEL 1: ", score1)
    print("MODEL 2: ", score2)
    print("MODEL 3: ", score3)
    print("MODEL 4: ", score4)
    print("\nAVG ACCURACY: ", (score1 + score2 + score3 + score4) / 4)
    print("MAX ACCURACY: ", max(score1, score2, score3, score4))


if __name__ == '__main__':
    get_data()
