import pandas as pd
import numpy as np
import math


def entropy(data):
    pos = 0
    neg = 0
    for value in data:
        if value == "YES":
            pos += 1
        else:
            neg += 1
    if pos == 0 or neg == 0:
        return 0
    else:
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        print("ent",-(p * math.log2(p) + n * math.log2(n)))
        return -(p * math.log2(p) + n * math.log2(n))


def info_gain(data, attr):
    uniq = np.unique(data[attr])
    gain = entropy(data["Infected"])
    for value in uniq:
        subdata = data[data[attr] == value]
        sub_e = entropy(subdata["Infected"])
        gain -= (len(subdata) / len(data)) * sub_e
    print(attr,gain)
    return gain


def id3(data, attrs, target):
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]
    elif len(attrs) == 0:
        return data[target].mode()[0]
    else:
        max_gain = -1
        best_attr = None
        for attr in attrs:
            gain = info_gain(data, attr)
            if gain > max_gain:
                max_gain = gain
                best_attr = attr
        
        tree = {best_attr: {}}
        remaining_attrs = attrs.copy()
        remaining_attrs.remove(best_attr)

        for value in np.unique(data[best_attr]):
            subdata = data[data[best_attr] == value]
            subtree = id3(subdata, remaining_attrs, target)
            tree[best_attr][value] = subtree

        return tree


def printTree(tree, indent=''):
    if isinstance(tree, dict):
        for attr, subtree in tree.items():
            print(indent + attr)
            printTree(subtree, indent + '\t')
    else:
        print(indent + '->', tree)


def predict(tree, data):
    for attr, subtree in tree.items():
        value = data[attr]
        if value in subtree:
            if isinstance(subtree[value], dict):
                return predict(subtree[value], data)
            else:
                return subtree[value]



df = pd.read_csv("CovidDataset.csv")

print("\nDataset: \n",df,'\n')

target = "Infected"
attributes = list(df.columns)
attributes.remove(target)

splited_df = []

splited_df = df[:5],df[5:10],df[10:14]


prediction = []
for df in splited_df:
    print(df)
    tree = id3(df, attributes, target)
    print("\nDecision Tree:\n")
    printTree(tree)
    sample = {"Fever": "NO", "Cough": "YES", "Breathing issues": "YES"}
    local_predict = predict(tree, sample)
    print("prediction: ",local_predict)
    prediction.append(local_predict)



if prediction.count('YES')>prediction.count('NO'):
    print("Final prediction: YES")
else:
    print("Final prediction: NO")


    




