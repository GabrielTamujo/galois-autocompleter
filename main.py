from flask import Flask, request, Response, abort
import json
from autocomplete import Autocomplete
import logging
logger = logging.getLogger(__name__)

app = Flask(__name__)

autocomplete = Autocomplete()


@app.route('/autocomplete', methods=['GET'])
def get(): return Response('', status=200)


@app.route('/autocomplete', methods=['POST'])
def post():
    input_text = request.get_json(force=True)['text']
    if input_text == '':
        abort(400, description="The input text cannot be null.")

    app.logger.info('Starting to predict.')
    result = autocomplete.predict(input_text)

    app.logger.info(f"Returning list of predictions: {result}")
    return Response(json.dumps({'result': result}), status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run('0.0.0.0', port=3030, debug=True, threaded = True)
