import re

class ShortSuggestion:
    def __init__(self, prediction):
        self.prediction = self.__get_firts_fragment(prediction)
        self.type = "SHORT"

    def __get_firts_fragment(self, prediction):
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