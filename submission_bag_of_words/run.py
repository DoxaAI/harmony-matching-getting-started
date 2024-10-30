import pathlib
import re
import sys
from typing import Any, Generator

import pandas as pd

directory = pathlib.Path(__file__).parent
sys.path.insert(0, str(directory.resolve()))

from competition import BaseEvaluator

#################################################################################
#                                                                               #
#   This file gets run when you submit your work for evaluation on the DOXA     #
#   AI platform. Modify the predict() method to implement your own strategy!    #
#                                                                               #
#################################################################################


re_tok = re.compile(r"(?i)([a-z']+)")


def to_bag_of_words(sentence):
    return set(re_tok.findall(sentence.lower()))


def jaccard_similarity(bag_of_words_1: set, bag_of_words_2: set):
    tokens_in_both_sentences = len(bag_of_words_1.intersection(bag_of_words_2))
    if tokens_in_both_sentences == 0:
        return 0

    tokens_in_any_sentence = len(bag_of_words_1.union(bag_of_words_2))
    return tokens_in_both_sentences / tokens_in_any_sentence


class Evaluator(BaseEvaluator):
    def predict(self, df: pd.DataFrame) -> Generator[int, Any, None]:
        """Write all the code you need to generate predictions for the test set here!

        Args:
            df (pd.DataFrame): This is a dataframe containing `sentence_1` and `sentence_`, just as in the training data

        Yields:
            Generator[int, Any, None]: For each pair of sentences in `df`, yield your similarity prediction,
                                       which should be an integer in the range [0, 100].
        """

        sentences = set(df["sentence_1"]) | set(df["sentence_2"])
        bags_of_words = {sentence: to_bag_of_words(sentence) for sentence in sentences}

        for _, row in df.iterrows():
            similarity = jaccard_similarity(
                bags_of_words[row["sentence_1"]],
                bags_of_words[row["sentence_2"]],
            )
            yield int(100 * similarity)


if __name__ == "__main__":
    Evaluator().run()
