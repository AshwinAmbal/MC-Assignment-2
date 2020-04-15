import requests
import json
import os


def get_data():
    dirList = [['fun', 'fun_PRACTICE_3_Nasim.json'], ['really', 'really_practice_2_gandavarapu.json'],
                 ['communicate', 'communicate_PRACTICE_1_Zaeifi.json'], ['buy', 'buy_PRACTISE_3_roongta.json'],
                 ['mother', 'mother_PRACTISE_3_sethi.json']]
    for row in dirList:
        with open(os.path.abspath('../JSON/{}/{}'.format(row[0], row[1]))) as fp:
            jsonFile = json.load(fp)
        headers = {'Content-type': 'application/json'}
        print(requests.post('http://127.0.0.1:9696/', data=json.dumps(jsonFile), headers=headers).json(), '\tAnswer = {}'.format(row[0]))


if __name__ == '__main__':
    get_data()
