import pathlib
import sys
from typing import Any, Generator

import pandas as pd

directory = pathlib.Path(__file__).parent
sys.path.insert(0, str(directory.resolve()))

from competition import BaseEvaluator
from sentence_transformers import SentenceTransformer

#################################################################################
#                                                                               #
#   This file gets run when you submit your work for evaluation on the DOXA     #
#   AI platform. Modify the predict() method to implement your own strategy!    #
#                                                                               #
#################################################################################

import re

re_tok = re.compile(r"(?i)([a-z']+)")

def to_bag_of_words(sentence):
    return set(re_tok.findall(sentence.lower()))

def jaccard_similarity(bag_of_words_1: set, bag_of_words_2: set):
    tokens_in_any_sentence = len(bag_of_words_1.union(bag_of_words_2))
    tokens_in_both_sentences = len(bag_of_words_1.intersection(bag_of_words_2))
    if tokens_in_both_sentences == 0:
        return 0
    ratio = tokens_in_both_sentences / tokens_in_any_sentence
    return ratio

class Evaluator(BaseEvaluator):
    def predict(self, df: pd.DataFrame) -> Generator[int, Any, None]:
        """Write all the code you need to generate predictions for the test set here!

        Args:
            df (pd.DataFrame): This is a dataframe containing `sentence_1` and `sentence_`, just as in the training data

        Yields:
            Generator[int, Any, None]: For each pair of sentences in `df`, yield your similarity prediction,
                                       which should be an integer in the range [0, 100].
        """

        #model = SentenceTransformer(str(directory / "model"))

        sentences = list(set(df["sentence_1"]) | set(df["sentence_2"]))

        bags_of_words = {}
        for sentence in sentences:
            bags_of_words[sentence] = to_bag_of_words(sentence)

        bags_of_words_1 = df["sentence_1"].map(bags_of_words)
        bags_of_words_2 = df["sentence_2"].map(bags_of_words)
        predictions = [0] * len(df)
        for idx in range(len(df)):
            predictions[idx] = jaccard_similarity(bags_of_words_1.iloc[idx], bags_of_words_2.iloc[idx])
        df["prediction"] = predictions
        df["prediction"] = (100 * df["prediction"]).apply(int).clip(0, 100)

        for _, row in df.iterrows():
            yield row["prediction"]


if __name__ == "__main__":
    Evaluator().run()
