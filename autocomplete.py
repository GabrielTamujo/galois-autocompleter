import gpt_2_simple as gpt2
import tensorflow as tf


class Autocomplete():

    def __init__(self):
        self.sess = start_tf_sess()
        self.graph = tf.get_default_graph()
        self.load_gpt2()

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
