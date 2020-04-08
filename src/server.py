from flask import Flask, jsonify
from cheroot.wsgi import Server as WSGIServer

app = Flask(__name__)


@app.route('/',  methods=['GET', 'POST'])
def hello_world():
    return jsonify({'Name': 'MC Assignment-2!!!'})

server = WSGIServer(bind_addr=('127.0.0.1', 9696), wsgi_app=app, numthreads=100)


if __name__ == '__main__':
    try:
        print("Serving on {}".format(server))
        server.start()
    except KeyboardInterrupt:
        server.stop()

