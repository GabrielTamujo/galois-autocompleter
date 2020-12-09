import re
from datetime import datetime

class Suggestions():

    def __init__(self, model_predictions):
        self.suggestions = []
        self.predictions_list = []
        self.__create_suggestions(model_predictions)

    def get_result(self):
        result = {
            "result": self.suggestions
        }
        print(result)
        return result
    
    def __create_suggestions(self, model_predictions):
        for prediction in model_predictions:
            self.__append_if_valid_suggestion(self.__new_long_suggestion(prediction))
            self.__append_if_valid_suggestion(self.__new_short_suggestion(prediction))

    def __append_if_valid_suggestion(self, suggestion):
        if self.__is_valid_prediction(suggestion['prediction']):
            self.predictions_list.append(suggestion['prediction'])
            self.suggestions.append(suggestion)
    
    def __is_valid_prediction(self, prediction):    
        return prediction and not prediction.isspace() and prediction not in self.predictions_list

    def __new_long_suggestion(self, prediction):
        return {
            "prediction": prediction,
            "type": "LONG"
        }

    def __new_short_suggestion(self, prediction):
        return {
            "prediction": self.__get_first_fragment(prediction),
            "type": "SHORT"
        }

    def __get_first_fragment(self, prediction):
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