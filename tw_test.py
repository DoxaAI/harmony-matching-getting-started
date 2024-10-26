from submission_bag_of_words.run import Evaluator
import pandas as pd

df = pd.DataFrame()
df["sentence_1"] = ["I feel sad", "I feel happy"]
df["sentence_2"] = ["I feel sad", "I feel sad"]

for result in Evaluator().predict(df):
    print (result)