import time
import json
from flask import Flask, Response, stream_with_context

app = Flask(__name__)

def get_suggestion():
    time.sleep(.1)
    return "Suggestion"

@app.route('/stream')
def stream():
    def gen():
        try:
            suggestions = []
            for i in range(20):
                suggestions.append(get_suggestion())
                print('Processing {}'.format(i))
                yield ""
            yield json.dumps({"Suggestions": ['a', 'b']})
        except GeneratorExit:
            print('closed')
    return Response(stream_with_context(gen()))

if __name__ == '__main__':
    app.run('0.0.0.0', port=3030, debug=True)