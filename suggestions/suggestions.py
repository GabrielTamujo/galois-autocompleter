from short_suggestion import ShortSuggestion
from long_suggestion import LongSuggestion

class Suggestions:
    
    def __init__(self, predictions_list):
        self.suggestions_list = []
        self.__create_suggestions(predictions_list)

    def __create_suggestions(self, predictions_list):
        for prediction in predictions_list:
            prediction = prediction.split('\n')[0]
            if self.__is_valid_prediction(prediction):
                self.__append_if_not_duplicate(ShortSuggestion(prediction))
                self.__append_if_not_duplicate(LongSuggestion(prediction))

    def __append_if_not_duplicate(self, new_suggestion):
        for suggestion in self.suggestions_list:
            if suggestion.prediction == new_suggestion.prediction:
                return
        self.suggestions_list.append(new_suggestion)


    def __is_valid_prediction(self, prediction):
        return prediction and not prediction.isspace()
