from flask import Flask, request, Response, abort
import gpt_2_simple as gpt2

import json

app = Flask(__name__)


@app.route('/', methods=['GET'])
def get(): return Response('', status=200, mimetype='application/json')


@app.route('/', methods=['POST'])
def post():
    input_text = request.get_json(force=True)['text']
    if input_text == '':
        abort(400, description="The input text cannot be null.")

    result = ['1', '2']
    
    # sess = gpt2.start_tf_sess()
    # gpt2.load_gpt2(sess, model_name='model')
    # result = gpt2.generate(sess,
    #                        seed=99,
    #                        nsamples=5,
    #                        batch_size=5,
    #                        length=8,
    #                        temperature=0,
    #                        top_k=10,
    #                        top_p=.85,
    #                        return_as_list=True)

    return Response(json.dumps({'result': result}), status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)
