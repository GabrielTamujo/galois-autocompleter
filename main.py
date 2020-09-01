from flask import Flask, request, Response, abort
import json
import gpt_2_simple as gpt2
import logging
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.logger.info('Starting to create tensorflow session.')
sess = gpt2.start_tf_sess(threads=1)

app.logger.info('Starting to load the GPT-2 model.')
gpt2.load_gpt2(sess,
               model_name='model',
               model_dir='',
               multi_gpu=True)


@app.route('/autocomplete', methods=['GET'])
def get(): return Response('', status=200)


@app.route('/autocomplete', methods=['POST'])
def post():
    request_body = request.get_json(force=True)

    app.logger.info('Starting to predict.')
    result = gpt2.generate(sess,
                           model_name='model',
                           model_dir='',
                           seed=request_body.get('seed', 99),
                           prefix="<|startoftext|>" + request_body.get('text', ''),
                           include_prefix=False,
                           truncate="<|endoftext|>",
                           nsamples=request_body.get('nsamples', 5),
                           batch_size=request_body.get('batch_size', 5),
                           length=request_body.get('length', 8),
                           top_k=request_body.get('top_k', 10),
                           top_p=request_body.get('top_p', .85),
                           temperature=request_body.get('temperature', 0.1),
                           return_as_list=True)

    app.logger.info(f"Returning list of predictions: {result}")
    return Response(json.dumps({'result': result}), status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run('0.0.0.0', port=3030, debug=True, threaded=True)
