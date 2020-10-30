from flask_restful import reqparse, abort, Api, Resource
from flask import Flask, jsonify, request, Response
from pymongo import MongoClient
from datetime import datetime
import tensorflow as tf
import numpy as np
import predictions
import logging
import encoder
import sample
import model
import json
import os

logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

def interact_model(model_name='model',
                   seed=99,
                   nsamples=2,
                   batch_size=2,
                   length=16,
                   temperature=0,
                   top_k=10,
                   top_p=.85,
                   models_dir=''):

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    gpu_options = tf.GPUOptions(allow_growth=True)

    config = tf.ConfigProto(intra_op_parallelism_threads=0, 
                            inter_op_parallelism_threads=0,
                            allow_soft_placement=True, 
                            gpu_options=gpu_options)

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, 
            length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, 
            top_k=top_k,
            top_p=top_p
        )
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        app = Flask(__name__)
        
        client = MongoClient('/galois/socket/mongodb-27017.sock')
        db = client.galois
        created_predictions = db.created_predictions
        accepted_predictions = db.accepted_predictions

        @app.route('/', methods=['GET'])
        def get(): return Response('', status=200)

        @app.route('/autocomplete', methods=['POST'])
        def create_predictions():
            body = request.get_json(force=True)
            if body['text'] == "":
                return

            text = body['text']
            text_array = body['text'].split("\n")
            total_lines = len(text_array)

            MAX_LINES_SUPPORTED = 30
            if total_lines > MAX_LINES_SUPPORTED:
                lines_discarded = total_lines - MAX_LINES_SUPPORTED
                text = '\n'.join(
                    text_array[lines_discarded: total_lines])

            context_tokens = enc.encode(text)

            predictions_list = []
            
            app.logger.info("Generating list of predictions.")
            for _ in range(nsamples // batch_size):
                feed_dict = {
                    context: [context_tokens for _ in range(batch_size)]}
                out = sess.run(output, feed_dict=feed_dict)[
                    :, len(context_tokens):]
                predictions_list = predictions.process(out, enc)
            
            if predictions_list: 
                app.logger.info("Saving list of predictions.")
                created_predictions.insert_one({
                    "predictions_list": predictions_list,
                    "datetime": str(datetime.now())
                })
                app.logger.info(f"Returning list of predictions: {predictions_list}")
                return Response(json.dumps({'result': predictions_list}), status=200, mimetype='application/json')
            
            return Response('No suggestions.', status=200, mimetype='application/json')

        @app.route('/acceptance', methods=['POST'])
        def save_accepted_prediction():
            app.logger.info("Persisting accepted prediction.")
            accepted_predictions.insert_one({
                "prediction": json.loads(request.data)['text'],
                "datetime": str(datetime.now())
            })
            return Response('', status=200)

        @app.route('/acceptance', methods=['GET'])
        def get_acceptance_report():
            response = {
                "predictions": created_predictions.count(),
                "acceptations": accepted_predictions.count()
            }
            return Response(json.dumps(response), status=200, mimetype='application/json')

        if __name__ == '__main__':
            app.run('0.0.0.0', port=3030, debug=True)


interact_model()
