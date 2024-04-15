from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from collections import Counter
import matplotlib.patches as patches

root_dir = os.path.dirname(os.path.abspath("data_process.py"))
toxic_data = pd.read_csv('hemolysis50_ave.csv')
k = 5
kf = KFold(n_splits = k, shuffle=True, random_state=123)
i = 0
for train_index, test_index in kf.split(toxic_data):
    train_path = '/home/jianxiu/Documents/amp_toxicity_regression/data/'+str(i)+'/train.csv'
    train_df = toxic_data.iloc[train_index]
    train_df.to_csv(train_path, index=False)

    test_path = '/home/jianxiu/Documents/amp_toxicity_regression/data/' + str(i) + '/test.csv'
    test_df = toxic_data.iloc[test_index]
    test_df.to_csv(test_path, index=False)
    i = i + 1