import logging

import flask
import fasttext
from flask_restful import reqparse
from flask import Flask, request, jsonify, Response

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# NOTE this import needs to happen after the logger is configured


# Initialize the Flask application
application = Flask(__name__)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('comment_path')


def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp


@application.route('/')
def index():
    return "Predict Wine Variety Based On Tasters' Reviews"
    
  
@application.route('/predict', methods=['GET'])
def predict():
    # Load in best model
    fasttext_model = fasttext.load_model("best_fasttext.bin")
    json_request = request.get_json()
    if not json_request:
        return Response("No json provided.", status=400)
    text = json_request['text']
    if text is None:
        return Response("No text provided.", status=400)
    else:
        prediction_result = fasttext_model.predict(text)

        result = {'Review':text, 'Label': str(prediction_result[0]), 'Probability': str(prediction_result[1])}
        return flask.jsonify(result)


if __name__ == '__main__':
    application.run(debug=True, port = '5000', use_reloader=True)

