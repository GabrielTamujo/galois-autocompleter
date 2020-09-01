import gpt_2_simple as gpt2
import tensorflow as tf


class Autocomplete():

    def __init__(self):
        self.sess = gpt2.start_tf_sess()
        self.graph = tf.get_default_graph()
        self.load_gpt2()

    def load_gpt2(self):
        with self.sess.as_default() as sess:
            with self.graph.as_default():
                gpt2.load_gpt2(sess,
                               model_name='model',
                               model_dir='',
                               multi_gpu=True)

    def predict(self, request_body):
        with self.sess.as_default() as sess:
            with self.graph.as_default():
                return gpt2.generate(sess,
                                     model_name='model',
                                     model_dir='',
                                     seed=request_body.get('seed', 99),
                                     prefix="<|startoftext|>" +
                                     request_body.get('text', ''),
                                     include_prefix=False,
                                     truncate="<|endoftext|>",
                                     nsamples=request_body.get('nsamples', 5),
                                     batch_size=request_body.get(
                                         'batch_size', 5),
                                     length=request_body.get('length', 8),
                                     top_k=request_body.get('top_k', 10),
                                     top_p=request_body.get('top_p', .85),
                                     temperature=request_body.get(
                                         'temperature', 0.1),
                                     return_as_list=True)
