import logging


# https://www.datacamp.com/community/tutorials/wordcloud-python
class WordCloud:

    def __init__(self):
        try:
            from wordcloud import WordCloud
        except ImportError:
            logging.error("-" * 100)
            logging.error("no wordcloud installation found. see https://pypi.org/project/wordcloud/")
            logging.error("-" * 100)
            pass

    def render(self, kata):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wordcloud = WordCloud().generate(["text"])

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()