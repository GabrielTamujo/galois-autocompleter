import gpt_2_simple as gpt2


class Autocomplete():

    def __init__(self):
        self.sess = gpt2.start_tf_sess()
        self.load_gpt2()

    def load_gpt2(self):
        with self.sess.as_default() as sess:
            gpt2.load_gpt2(sess,
                           model_name='model',
                           model_dir='',
                           multi_gpu=True)

    def predict(self, input_text):
        with self.sess.as_default() as sess:
            return gpt2.generate(sess,
                                 model_name='model',
                                 model_dir='',
                                 prefix=input_text,
                                 include_prefix=False,
                                 nsamples=5,
                                 batch_size=5,
                                 length=8,
                                 temperature=0.5,
                                 return_as_list=True)
