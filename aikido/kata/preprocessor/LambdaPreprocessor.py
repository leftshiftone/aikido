from aikido.__api__ import Preprocessor


class LambdaPreprocessor(Preprocessor):

    def __init__(self, mapper):
        self.mapper = mapper

    def preprocess(self, text: str):
        return self.mapper(text)