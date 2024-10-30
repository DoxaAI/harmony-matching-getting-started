import pathlib
import sys
from typing import Any, Generator

directory = pathlib.Path(__file__).parent
sys.path.insert(0, str(directory.resolve()))

import pandas as pd
from competition import BaseEvaluator
from sentence_transformers import SentenceTransformer

#################################################################################
#                                                                               #
#   This file gets run when you submit your work for evaluation on the DOXA     #
#   AI platform. Modify the predict() method to implement your own strategy!    #
#                                                                               #
#################################################################################


class Evaluator(BaseEvaluator):
    def predict(self, df: pd.DataFrame) -> Generator[int, Any, None]:
        """Write all the code you need to generate predictions for the test set here!

        Args:
            df (pd.DataFrame): This is a dataframe containing `sentence_1` and `sentence_2`, just as in the training data

        Yields:
            Generator[int, Any, None]: For each pair of sentences in `df`, yield your similarity prediction,
                                       which should be an integer in the range [0, 100].
        """

        model = SentenceTransformer(str(directory / "model"))

        sentences = list(set(df["sentence_1"]) | set(df["sentence_2"]))
        embeddings = {
            sentence: embedding
            for sentence, embedding in zip(
                sentences, model.encode(sentences, batch_size=16)
            )
        }

        df["prediction"] = model.similarity_pairwise(
            df["sentence_1"].map(embeddings),
            df["sentence_2"].map(embeddings),
        )
        df["prediction"] = (100 * df["prediction"]).apply(int).clip(0, 100)

        for _, row in df.iterrows():
            yield row["prediction"]


if __name__ == "__main__":
    Evaluator().run()
