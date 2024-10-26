import os
from typing import Any, Generator

import pandas as pd


class BaseEvaluator:
    def predict(self, df: pd.DataFrame) -> Generator[int, Any, None]:
        raise NotImplementedError

    def run(self):
        stream_directory = os.environ.get("DOXA_STREAMS")

        in_file = f"{stream_directory}/in" if stream_directory else "train.csv"
        out_file = f"{stream_directory}/out" if stream_directory else "predictions.csv"

        with (
            open(in_file, "r", encoding="utf8") as r,
            open(out_file, "w") as w,
        ):
            w.write(f"OK\n")
            w.flush()

            df = pd.read_csv(r)
            for prediction in self.predict(df):
                w.write(f"{prediction}\n")
                w.flush()
