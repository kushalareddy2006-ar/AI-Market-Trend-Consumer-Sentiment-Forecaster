import pandas as pd

df = pd.read_csv("combined_cleaned_data.csv")

# kee only 30% of the data
KEEP_RATIO = 0.3


df_reduced =(
    df.groupby(['product', 'rating'], group_keys=False)   #grouiping product and rating
    .apply(lambda x:x.sample(n=max(1, int(len(x)*KEEP_RATIO)), random_state=42))
    .reset_index(drop=True)    #resetting index
)

print(f"Original data size: {len(df)}")
print(f"Reduced data size: {len(df_reduced)}")

df_reduced.to_csv("reduced_combined_cleaned_data.csv", index=False)