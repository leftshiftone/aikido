import string

from aikido.__api__ import Preprocessor


class PunctuationPreprocessor(Preprocessor):
    """
    Preprocessor implementation which removes punctuations from the text.
    """

    def __init__(self):
        pass

    def preprocess(self, text: str):
        return text.translate(str.maketrans('', '', string.punctuation))