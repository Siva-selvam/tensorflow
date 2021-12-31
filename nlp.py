import pandas as pd
train_df = pd.read_csv("/Users/apple/Downloads/tensorflow notes/train.csv")
test_df = pd.read_csv("/Users/apple/Downloads/tensorflow notes/test.csv")
train_df.head()
train_df_shuffled = train_df.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility
train_df_shuffled.head()
train_df.target.value_counts()
