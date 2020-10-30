import re

def process(model_outputs, enc, batch_size):
    predictions = []
    for i in range(batch_size):
        prediction = enc.decode(output[i])
        prediction = filter_noise(prediction)
        prediction = remove_new_line(prediction)
        prediction = escape_spaces_from_beggining(prediction)
        if is_valid_prediction(prediction) and prediction not in predictions:
            predictions.append({
                "prediction": str(prediction),
                "type": "MULTIPLE_TOKENS"
                })
            first_token = get_first_token(str(prediction))
            if is_valid_prediction(prediction) and first_token not in predictions:
                predictions.append({
                    "prediction": first_token,
                    "type": "SINGLE_TOKEN"
                    })

def is_valid_prediction(prediction):
    return prediction and not prediction.isspace()

def filter_noise(prediction):
    return prediction.replace("▄", "").replace("█", "")

def remove_new_line(prediction):
    return prediction.split('\n')[0]

def escape_spaces_from_beggining(prediction):
    matrix = prediction.split(' ')
    while matrix !=[] and not matrix[0]:
        del matrix[0]
    return ' '.join(matrix)

def get_first_token(prediction):
    delimiters = [' ',
                '_',
                '.',
                '(',
                ')',
                '{',
                '}',
                '[',
                ']',
                ',',
                ':',
                '\'',
                '"',
                '=',
                '<',
                '>',
                '/',
                '\\',
                '+',
                '-',
                '|',
                '&',
                '*',
                '%',
                '=',
                '#',
                '!']
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, prediction)[0]