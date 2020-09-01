from flask_restful import Resource, Api, abort
from flask import Flask, jsonify, request, Response
import json

import tensorflow as tf
import gpt_2_simple as gpt2

app = Flask(__name__)

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(intra_op_parallelism_threads=0,
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        gpu_options=gpu_options)

app.logger.info('Starting new tensorflow session.')
with tf.Session(graph=tf.Graph(), config=config) as sess:

    app.logger.info('Starting to load GPT-2 model.')
    gpt2.load_gpt2(sess,
                   model_name='model',
                   model_dir='',
                   multi_gpu=True)

    class Autocomplete(Resource):

        def get(self): return ''

        def post(self):
            input_text = request.get_json(force=True)['text']
            if input_text == '':
                abort(400, description="The input text cannot be null.")

            app.logger.info('Starting to generate samples')
            result = gpt2.generate(sess,
                                   model_name='model',
                                   model_dir='',
                                   prefix=input_text,
                                   include_prefix=False,
                                   nsamples=5,
                                   batch_size=5,
                                   length=8,
                                   temperature=0.5,
                                   return_as_list=True)

            app.logger.info(f"Returning list of predictions: {result}")
            return Response(json.dumps({'result': result}), status=200, mimetype='application/json')


app = Flask(__name__)
api = Api(app)
api.add_resource(Autocomplete, '/autocomplete')

if __name__ == '__main__':
    app.run('0.0.0.0', port=3030, debug=True)
