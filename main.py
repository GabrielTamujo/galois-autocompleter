from flask_restful import Resource, Api, abort
from flask import Flask, jsonify, request, Response
import json

import gpt_2_simple as gpt2

app = Flask(__name__)

# with gpt2.start_tf_sess() as sess:

#     gpt2.load_gpt2(sess,
#                    model_name='model',
#                    model_dir='')


class Autocomplete(Resource):

    def get(self): return ''

    def post(self):
        input_text = request.get_json(force=True)['text']
        if input_text == '':
            abort(400, description="The input text cannot be null.")

        sess = gpt2.start_tf_sess()

        gpt2.load_gpt2(sess,
                       model_name='model',
                       model_dir='')

        result = gpt2.generate(sess,
                               model_name='model',
                               model_dir='',
                               prefix=input_text,
                               nsamples=5,
                               batch_size=5,
                               length=8,
                               temperature=0.5,
                            #    top_k=10,
                            #    top_p=.85
                               )
        app.logger.info(f"Returning list of predictions: {result}")
        return Response(json.dumps({'result': result}), status=200, mimetype='application/json')


app = Flask(__name__)
api = Api(app)
api.add_resource(Autocomplete, '/autocomplete')

if __name__ == '__main__':
    app.run('0.0.0.0', port=3030, debug=True)
