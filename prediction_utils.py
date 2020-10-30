import re

def process(prediction):
    prediction = filter_noise(prediction)
    prediction = remove_new_line(prediction)
    prediction = escape_spaces_from_beggining(prediction)
    return prediction

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