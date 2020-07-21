from aikido.__api__ import Preprocessor


class EnsemblePreprocessor(Preprocessor):

    def __init__(self, preprocessors: [Preprocessor]):
        self.preprocessors = preprocessors

    def preprocess(self, text: str):
        for preprocessor in self.preprocessors:
            text = preprocessor.preprocess(text)
        return text
