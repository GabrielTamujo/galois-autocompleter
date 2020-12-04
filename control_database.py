from pymongo import MongoClient
from datetime import datetime

class ControlDatabase():

    def __init__(self):
        self.client = MongoClient('/galois/socket/mongodb-27017.sock')
        self.created_suggestions = client.galois.created_suggestions
        self.accepted_suggestions = client.galois.accepted_suggestions

    def save_suggestions(self, suggestions):
        self.created_suggestions.insert_one({
            "suggestions_list": suggestions,
            "datetime": str(datetime.now())
        })

    def save_accepted_suggestion(self, suggestion):
        self.created_suggestions.insert_one({
            "suggestion": suggestion,
            "datetime": str(datetime.now())
        })