import gpt_2_simple as gpt2
from gpt_2_simple import encoder, model, sample
import tensorflow as tf
import numpy as np
import json
import os

class Autocomplete():
    
    model_name='model'
    models_dir=''
    seed=99
    nsamples=5
    batch_size=5
    length=8 
    temperature=0
    top_k=10
    top_p=.85
    
    def __init__():
        self.sess = self.start_tf_sess()
        self.graph = tf.get_default_graph()
        self.encoder = encoder.get_encoder()
        self.hparams = model.default_hparams()
        self.override_hparams()
        self.load_gpt2()
    
    def override_hparams(self):
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))
    
    def start_tf_sess(self):
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True, 
                        gpu_options=gpu_options)
        return tf.compat.v1.Session(config=config)
    
    def load_gpt2(self):
        with self.sess.as_default() as sess:
            with self.graph.as_default():
                if model_name:
                    checkpoint_path = os.path.join(model_dir, model_name)
                else:
                    checkpoint_path = os.path.join(checkpoint_dir, run_name)

                context = tf.compat.v1.placeholder(tf.int32, [1, None])

                gpus = gpt2.get_available_gpus()

                self.output = sample.sample_sequence(
                    hparams=self.hparams, 
                    length=length,
                    context=context,
                    batch_size=batch_size,
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p
                )

                if checkpoint=='latest':
                    ckpt = tf.train.latest_checkpoint(checkpoint_path)
                else:
                    ckpt = os.path.join(checkpoint_path,checkpoint)

                saver = tf.compat.v1.train.Saver(allow_empty=True)
                sess.run(tf.compat.v1.global_variables_initializer())

                if model_name:
                    print('Loading pretrained model', ckpt)
                else:
                    print('Loading checkpoint', ckpt)
                saver.restore(sess, ckpt)

    def predict(self, body):
        if body['text'] == "": return
        with self.sess.as_default() as sess:
            with self.graph.as_default():
                context_tokens = encoder.encode(request_body.get('text', ''))
                generated = 0
                predictions = []

                for _ in range(nsamples // batch_size):

                    feed_dict = {context: [context_tokens for _ in range(batch_size)]}
                    out = sess.run(self.output, feed_dict=feed_dict)[:, len(context_tokens):]

                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
                        predictions.append(str(text))              
                
                return predictions

    
    
    
    # def predict(self, request_body):
    #     with self.sess.as_default() as sess:
    #         with self.graph.as_default():
    #             return gpt2.generate(sess,
    #                                  model_name='model',
    #                                  model_dir='',
    #                                  seed=request_body.get('seed', 99),
    #                                  prefix="<|startoftext|>" +
    #                                  request_body.get('text', ''),
    #                                  include_prefix=False,
    #                                  truncate="<|endoftext|>",
    #                                  nsamples=request_body.get('nsamples', 5),
    #                                  batch_size=request_body.get(
    #                                      'batch_size', 5),
    #                                  length=request_body.get('length', 8),
    #                                  top_k=request_body.get('top_k', 10),
    #                                  top_p=request_body.get('top_p', .85),
    #                                  temperature=request_body.get(
    #                                      'temperature', 0.1),
    #                                  return_as_list=True)
