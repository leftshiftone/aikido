import logging
import re

from aikido.__api__ import Preprocessor

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')


class SpacyPreprocessor(Preprocessor):

    def __init__(self, model: str = "de_core_news_sm", stem: bool = False):
        try:
            import spacy

            self.stem = stem
            self.NLP = spacy.load(model)
            self.NLP.tokenizer = self.custom_tokenizer(self.NLP)
        except ImportError:
            logging.error("-" * 100)
            logging.error("no spacy installation found. see https://spacy.io/usage")
            logging.error("-" * 100)
            pass

    def preprocess(self, text: str):
        tokens = [tok.lemma_ if self.stem else tok.text for tok in self.NLP.tokenizer(text)]
        tokens = filter(lambda x: not self.NLP.vocab[x].is_punct and not len(x) < 2, tokens)

        return " ".join(tokens)

    def custom_tokenizer(self, nlp):
        from spacy.tokenizer import Tokenizer
        from spacy.util import compile_prefix_regex, compile_suffix_regex
        infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~\-/]''')
        prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
        suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

        return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         token_match=None)