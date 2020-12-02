import torch
from config import Config
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from suggestions.suggestions import Suggestions

class PythonPredictor:

    def __init__(self, config):
        model_name_or_path = config.get("model_name_or_path", "distilgpt2")
        self.device = self.__get_device()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(self.device)
        self.config = Config(config, self.model.config.max_position_embeddings)

    def predict(self, payload):
        input_text = payload["text"]
        input_text = input_text[max(len(input_text) - self.config.MAX_INPUT_TEXT_LENGTH, 0):]
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        input_ids = input_ids[max(len(input_ids) - self.config.MAX_INPUT_TOKENS_LENGTH, 0):]
        input_ids_length = len(input_ids)

        sample_outputs = self.model.generate(
            input_ids=input_ids,
            top_p=self.config.TOP_P,
            top_k=self.config.TOP_K,
            temperature=self.config.TEMPERATURE, 
            max_length=input_ids_length + self.config.NUM_TOKENS_TO_PREDICT, 
            num_return_sequences=self.config.NUM_RETURN_SEQUENCES,
            do_sample=True,
        )

        predictions_list = []
        for sample_output in sample_outputs:
            predictions_list.append(self.tokenizer.decode(sample_output[input_ids_length:].tolist(), skip_special_tokens=True))

        return Suggestions(predictions_list)

    def __get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {device}")
        return device
    

            





