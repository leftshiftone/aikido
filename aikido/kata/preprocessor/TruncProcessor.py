from typing import List

from aikido.__api__ import Preprocessor


class TruncProcessor(Preprocessor):

    def __init__(self, text_list:List[str], prefix:bool = True):
        self.text_list = text_list
        self.prefix = prefix

    def preprocess(self, text: str):
        for e in self.text_list:
            index = text.find(e)
            if index >= 0 and self.prefix:
                return text[index + len(e):]
            if index >= 0 and not self.prefix:
                return text[:index]

        return text
    