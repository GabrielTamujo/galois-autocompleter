from flask_restful import Resource, abort
from flask import jsonify, request, Response
from flask import current_app as app
import json

import gpt_2_simple as gpt2

import logging
logger = logging.getLogger(__name__)

class Autocomplete(Resource):

    def __init__(self, sess):
        self.sess = sess

    def get(self): return ''

    def post(self):
        input_text = request.get_json(force=True)['text']
        if input_text == '':
            abort(400, description="The input text cannot be null.")

        result = gpt2.generate(self.sess,
                               model_name='model',
                               model_dir='',
                               seed=99,
                               nsamples=5,
                               batch_size=5,
                               length=8,
                               temperature=0,
                               top_k=10,
                               top_p=.85,
                               return_as_list=True)
        app.logger.info(f"Returning list of predictions: {result}")
        return Response(json.dumps({'result': result}), status=200, mimetype='application/json')