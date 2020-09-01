from flask_restful import Api
from flask import Flask
from autocomplete import Autocomplete
import gpt_2_simple as gpt2

app = Flask(__name__)

with gpt2.start_tf_sess() as sess:

    gpt2.load_gpt2(sess,
                   model_name='model',
                   model_dir='')

    app = Flask(__name__)
    api = Api(app)
    api.add_resource(Autocomplete(sess), '/autocomplete')

    if __name__ == '__main__':
        app.run('0.0.0.0', port=3030, debug=True)
