# -*- coding: utf-8 -*

import pandas as pd
import os,sys
import decision_tree

with open(os.path.abspath('.')+"/data.csv", mode='r',encoding='utf-8') as data_file:
    df = pd.read_csv(data_file)

root = decision_tree.TreeGenerate(df)
accuracy_scores = []
n = len(df.index)
k = 5
for i in range(k):
    m = int(n / k)
    test = []
    for j in range(i * m, i * m + m):
        test.append(j)

    df_train = df.drop(test)
    df_test = df.iloc[test]
    root = decision_tree.TreeGenerate(df_train)  # generate the tree

    # test the accuracy
    pred_true = 0
    for i in df_test.index:
        label = decision_tree.Predict(root, df[df.index == i])
        if label == df_test[df_test.columns[-1]][i]:
            pred_true += 1

    accuracy = pred_true / len(df_test.index)
    accuracy_scores.append(accuracy)

# print the prediction accuracy result
accuracy_sum = 0
print("accuracy: ", end="")
for i in range(k):
    print("%.3f  " % accuracy_scores[i], end="")
    accuracy_sum += accuracy_scores[i]
print("\naverage accuracy: %.3f" % (accuracy_sum / k))

# dicision tree visualization using pydotplus.graphviz
import os
os.environ["PATH"] += os.pathsep + 'D:/software/graphviz-2.38/release/bin'
decision_tree.DrawPNG(root, "decision_tree_ID3.png")