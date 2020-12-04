import re
from datetime import datetime

#TODO: this code works, but is terrible
def create_suggestions(model_predictions):
    print(model_predictions)
    
    suggestions = []
    predictions_list = []

    for prediction in model_predictions:
        if is_valid_prediction(prediction) and prediction not in predictions_list:
            suggestions.append(new_long_suggestion(prediction))
            predictions_list.append(prediction)

            first_fragment = get_first_fragment(prediction)
            if is_valid_prediction(first_fragment) and first_fragment not in predictions_list:
                suggestions.append(new_short_suggestion(first_fragment))
                predictions_list.append(first_fragment)
    return {
        "result": suggestions
    }

def is_valid_prediction(prediction):    
    return prediction and not prediction.isspace()

def new_long_suggestion(prediction):
    return {
        "prediction": prediction,
        "type": "LONG"
    }

def new_short_suggestion(prediction):
    return {
        "prediction": prediction,
        "type": "SHORT"
    }

def get_first_fragment(prediction):
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