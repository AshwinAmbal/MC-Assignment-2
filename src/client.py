import requests
import json
import os


def get_data():
    with open(os.path.abspath('../JSON/buy/buy_PRACTISE_3_sethi.json')) as fp:
        jsonFile = json.load(fp)
    headers = {'Content-type': 'application/json'}
    print(requests.post('http://127.0.0.1:9696/', data=json.dumps(jsonFile), headers=headers).json())


if __name__ == '__main__':
    get_data()
