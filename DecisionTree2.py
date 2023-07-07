import numpy as np
import pandas as pd

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    #print(target_col,entropy_val
    print(entropy_val)
    return entropy_val

def InfoGain(data, split_attribute_name, target_name="yes"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    print(data,"\n",split_attribute_name,target_name,information_gain,"\n")
    return information_gain

def ID3(data, original_data, features, target_attribute_name="Infected", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data)==0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class 
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

# Example usage:
data = pd.read_csv('CovidDataset.csv')  # Replace with your dataset
target_attribute = 'Infected'
features = list(data.columns)
features.remove(target_attribute)
tree = ID3(data, data, features)
print(tree)
