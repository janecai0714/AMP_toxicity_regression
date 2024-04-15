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

data_df_ave = pd.read_csv("/home/jianxiu/Documents/amp_toxicity_regression/data_analysis/hemolysis50_ave.csv")
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
p2_bins = np.arange(-4, 1, 0.5)
plt.hist(data_df_ave['pHD50'], color="lightblue", edgecolor="black", bins=p2_bins)
plt.xlabel("pHD50", size=15)
plt.ylabel("Frequency",  size=15)
xtick = np.arange(-4, 1, 0.5)
plt.xticks(fontsize=12)
ytick = np.arange(0, 1100, 100)
plt.yticks(ytick, fontsize=12)
plt.xlim([-4, 1])
plt.ylim([0, 1000])
train_blue_patch = patches.Patch(color='lightblue', label='whole set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[train_blue_patch])
plt.savefig("/home/jianxiu/Documents/amp_toxicity_regression/data_analysis/whole_pHD50_dist.svg")

train_path = '/home/jianxiu/Documents/amp_toxicity_regression/data/0/train.csv'
test_path  = '/home/jianxiu/Documents/amp_toxicity_regression/data/0/test.csv'

# train draw length density
train = pd.read_csv(train_path)
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
p1_bins = np.arange(0, 55, 5)
plt.hist(train['LENGTH'], color="lightblue", edgecolor="black", bins=p1_bins)
xtick = np.arange(0, 55, 5)
plt.xticks(xtick, fontsize=12)
plt.xlim([0, 55])
ytick = np.arange(0, 1100, 100)
plt.yticks(ytick, fontsize=12)
plt.ylim([0, 1000])
plt.ylabel("Frequency",  size=15)
plt.xlabel("Length", size=15)
train_blue_patch = patches.Patch(color='lightblue', label='Train set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[train_blue_patch])
plt.savefig("/home/jianxiu/Documents/amp_toxicity_regression/data_analysis/train_len_hist.svg")
plt.close()

p2_bins = np.arange(-4, 1, 0.5)
plt.hist(train['pHD50'], color="lightblue", edgecolor="black", bins=p2_bins)
plt.xlabel("pHD50 (-log μM)", size=15)
plt.ylabel("Frequency",  size=15)
xtick = np.arange(-4, 1, 0.5)
plt.xticks(fontsize=12)
ytick = np.arange(0, 1100, 100)
plt.yticks(ytick, fontsize=12)
plt.xlim([-4, 1])
plt.ylim([0, 1000])
train_blue_patch = patches.Patch(color='lightblue', label='Train set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[train_blue_patch])
plt.savefig("/home/jianxiu/Documents/amp_toxicity_regression/data_analysis/train_pHD50_dist.svg")

train_amino_acids =" ".join(train["SEQUENCE_space"]).split()
train_aa = pd.Series(train_amino_acids).value_counts(sort=False).reset_index()
train_aa = train_aa.rename(columns={"index" : "aa", 0 : "count"})
train_aa = train_aa.sort_values("aa")
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
plt.bar(train_aa["aa"], train_aa["count"] , color="lightblue", edgecolor="black")
plt.xlabel("Amino acids", size=15) #, fontweight='bold'
plt.ylabel("Frequency",  size=15)
plt.xticks(fontsize=10) # , rotation=90)
ytick = np.arange(0, 7000, 1000)
plt.yticks(fontsize=10)
plt.ylim([0, 7000])
train_blue_patch = patches.Patch(color='lightblue', label='Train set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[train_blue_patch])
plt.savefig("/home/jianxiu/Documents/amp_toxicity_regression/data_analysis/train_aa_hist.svg")
plt.close()

# test set
test_color = "dodgerblue"
test = pd.read_csv(test_path)
plt.hist(test['LENGTH'], color=test_color, edgecolor="black", bins=p1_bins)
xtick = np.arange(0, 55, 5)
plt.xticks(xtick, fontsize=12)
plt.xlim([0, 55])
# ytick = np.arange(0, 1100, 100)
plt.yticks(fontsize=12)
plt.ylim([0, 200])
plt.ylabel("Frequency",  size=15)
plt.xlabel("Length", size=15)
test_green_patch = patches.Patch(color=test_color, label='Test set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[test_green_patch])
plt.savefig("/home/jianxiu/Documents/amp_toxicity_regression/data_analysis/test_len_hist.svg")
plt.close()

test_pmic = test["pHD50"]
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
p2_bins = np.arange(-4, 1, 0.5)
plt.hist(test_pmic, color=test_color, edgecolor="black", bins=p2_bins) # aquamarine
plt.xlabel("pHD50 (-log μM)", size=15)
plt.ylabel("Frequency",  size=15)
xtick = np.arange(-4, 1, 0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([-4, 1])
plt.ylim([0, 200])
test_green_patch = patches.Patch(color=test_color, label='Test set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[test_green_patch])
plt.savefig("/home/jianxiu/Documents/amp_toxicity_regression/data_analysis/test_pHD50_hist.svg")
plt.close()

test_amino_acids =" ".join(test["SEQUENCE_space"]).split()
test_aa = pd.Series(test_amino_acids).value_counts(sort=False).reset_index()
test_aa = test_aa.rename(columns={"index" : "aa", 0 : "count"})
test_aa = test_aa.sort_values("aa")
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
plt.bar(test_aa["aa"], test_aa["count"] , color=test_color, edgecolor="black")
plt.xlabel("Amino acids", size=15) #, fontweight='bold'
plt.ylabel("Frequency",  size=15)
plt.xticks(fontsize=10) # , rotation=90)
ytick = np.arange(0, 1600, 100)
plt.yticks(ytick, fontsize=10)
#plt.ylim([0, 1500])
test_green_patch = patches.Patch(color=test_color, label='Test set')
plt.rcParams["legend.fontsize"] = 15
plt.legend(handles=[test_green_patch])
plt.savefig("/home/jianxiu/Documents/amp_toxicity_regression/data_analysis/test_aa_hist.svg")
plt.close()