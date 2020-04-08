import requests


def get_data():
    print(requests.get('http://127.0.0.1:9696/').json())


if __name__ == '__main__':
    get_data()
