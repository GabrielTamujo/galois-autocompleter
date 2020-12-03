import torch
from config import Config
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from suggestions import create_suggestions

class PythonPredictor:

    def __init__(self, config):
        model_name_or_path = config.get("model_name_or_path", "distilgpt2")
        self.device = self.__get_device()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path, add_prefix_space=True)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(self.device)
        self.config = Config(config, self.model.config.max_position_embeddings)

    def predict(self, payload):
        input_text = payload["text"]
        input_text = input_text[max(len(input_text) - self.config.MAX_INPUT_TEXT_LENGTH, 0):]
        print(input_text)
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        input_ids_length = len(input_ids[0])
        input_ids = input_ids[max(input_ids_length - self.config.MAX_INPUT_TOKENS_LENGTH, 0):]

        #TODO: verify if bad_words_ids works to avoid '\n'
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
            predicted_sequence = sample_output[input_ids_length:].tolist()
            predictions_list.append(self.tokenizer.decode(predicted_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True))

        return create_suggestions(predictions_list)

    def __get_device(self):
        #TODO: get the available device with less memory allocated
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {device}")
        return device
    

            





