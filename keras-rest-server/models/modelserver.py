from flask import Flask, json
from flask.views import MethodView
# from flask.ext.cors import cross_origin
from flask_cors import cross_origin
from flask import Flask, request, jsonify, g
from modelcreator import Predictor
from gevent.pywsgi import WSGIServer
import numpy as np
import settings

app = Flask(__name__)
predictor = None


class ModelLoader(MethodView):
    def __init__(self):
        pass

    def post(self):
        content = request.get_json()
        ## Uncomment below line if sending url and comment the one below that
        # X_input = str(content['input'])
        X_input = content
        pred_val = predictor.predict(X_input=X_input)
        pred_val = pred_val.tolist()
        return json.jsonify({'pred_val': pred_val})


def initialize_models(model_path):
    global predictor
    predictor = Predictor(model_path)
    


def run(host='0.0.0.0', port=7171):
    """
    run a WSGI server using gevent
    """
    app.add_url_rule('/predict', view_func=ModelLoader.as_view('predict'))
    print 'running server http://{0}'.format(host + ':' + str(port))
    WSGIServer((host, port), app).serve_forever()
