class Config:
    def __init__(self, config, model_max_length):
        self.NUM_TOKENS_TO_PREDICT = config.get("num_tokens_to_predict", 16)
        self.MAX_INPUT_TOKENS_LENGTH = model_max_length - self.NUM_TOKENS_TO_PREDICT
        self.MAX_INPUT_TEXT_LENGTH = config.get("max_input_text_length", 1000)
        self.NUM_RETURN_SEQUENCES = config.get("num_return_sequences", 2)
        self.TOP_K = config.get("top_k", 50)
        self.TOP_P = config.get("top_p", 0.85)
        self.TEMPERATURE = config.get("temperature", 0.5)
        self.END_OF_LINE_TOKEN_ID = config.get("end_of_line_token_id", 50258)