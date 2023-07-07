import pandas as pd 
data = pd.read_csv("PreProcessingDataset.csv")
df = pd.DataFrame(data)
print("DataFrame: \n",df)
print("\nDuplicates: \n",df.duplicated())
df1 = df.drop_duplicates()
print("DataFrame1: \n",df1)
single_valued_columns = []
for col in df1.columns:
    if(len(df1[col].dropna().unique()) == 1):
        single_valued_columns.append(col)

print("\nSingle Valued Columns: ",single_valued_columns)

df2 = df1.drop(single_valued_columns,axis=1)
print("\nData Frame2: \n",df2)