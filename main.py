from flask_restful import Resource, Api, abort
from flask import Flask, jsonify, request, Response
import json

import gpt_2_simple as gpt2

app = Flask(__name__)

# app.logger.info('Starting new tensorflow session.')
# with gpt2.start_tf_sess() as sess:

class Autocomplete(Resource):

    def get(self): return ''

    def post(self):
        input_text = request.get_json(force=True)['text']
        if input_text == '':
            abort(400, description="The input text cannot be null.")

        app.logger.info('Starting new tensorflow session.')
        sess = gpt2.start_tf_sess()

        app.logger.info('Starting to load GPT-2 model')
        gpt2.load_gpt2(sess,
                       model_name='model',
                       model_dir='',
                       multi_gpu=True)

        app.logger.info('Starting to generate samples')
        result = gpt2.generate(sess,
                               model_name='model',
                               model_dir='',
                               seed=99,
                               prefix=input_text,
                               include_prefix=False,
                               nsamples=5,
                               batch_size=5,
                               length=8,
                               temperature=0,
                               top_k=10,
                               top_p=.85,
                               return_as_list=True)

        app.logger.info(f"Returning list of predictions: {result}")
        return Response(json.dumps({'result': result}), status=200, mimetype='application/json')


app = Flask(__name__)
api = Api(app)
api.add_resource(Autocomplete, '/autocomplete')

if __name__ == '__main__':
    app.run('0.0.0.0', port=3030, debug=True)
