import pandas as pd

# Change this to import a different submission evaluator
from submission.run import Evaluator


def main():
    df = pd.DataFrame(
        data=[
            ("I feel sad", "I feel sad"),
            ("I feel happy", "I feel sad"),
        ],
        columns=["sentence_1", "sentence_2"],
    )

    for (i, row), prediction in zip(df.iterrows(), Evaluator().predict(df)):
        print(f"{i:<4} {row['sentence_1']:<24} {row['sentence_2']:<24} {prediction}")


if __name__ == "__main__":
    main()
