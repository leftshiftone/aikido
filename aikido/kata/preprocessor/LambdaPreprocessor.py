from aikido.__api__ import Preprocessor


class LambdaPreprocessor(Preprocessor):
    """
    Preprocessor implementation which uses the given lambda for preprocessing.
    """

    def __init__(self, mapper):
        self.mapper = mapper

    def preprocess(self, text: str):
        return self.mapper(text)
