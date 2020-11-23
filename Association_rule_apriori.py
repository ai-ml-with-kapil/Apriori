# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

#Reading DataSet
df = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

#List of components 
item_list = []
for i in range(0,df.shape[0]):
    item_list.append([df.values[i,j] for j in range(0,df.shape[1])])


#removing nan
final_list = []
for i in range(0, len(item_list)):
    if np.nan in item_list[i]:
        final_list.append(list(set(item_list[i]) ^ set([np.nan])))

#Training dataset
rules = apriori(transactions=final_list, min_support = (3*7/df.shape[0]), min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

#Storing all rules
results = list(rules)

# Preparing dataframe for easy understanding 
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

#Printing top 5 suggestions, sorted with lift.
print(resultsinDataFrame.nlargest(n = 5, columns = 'Lift'))