from typing import List, Any, Union, Tuple

import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import seaborn as sns

headlines = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "target"]
targets = ["not_recom", "recommend", "very_recom", "priority", "spec_prior"]
features = headlines[:-1]
label = headlines[-1]

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data', names=headers)
df.head()

df.parents = df.parents.astype("category", categories=["usual", "pretentious", "great_pret"], ordered=True)
df.has_nurs = df.has_nurs.astype("category", categories=["proper", "less_proper", "improper", "critical", "very_crit"], ordered=True)
df.form = df.form.astype("category", categories=["complete", "completed", "incomplete", "foster"], ordered=True)
df.children = df.children.astype("category", categories=["1", "2", "3", "more"], ordered=True)
df.housing = df.housing.astype("category", categories=["convenient", "less_conv", "critical"], ordered=True)
df.finance = df.finance.astype("category", categories=["convenient", "inconv"], ordered=True)
df.social = df.social.astype("category", categories=["nonprob", "slightly_prob", "problematic"], ordered=True)
df.health = df.health.astype("category", categories=["recommended", "priority", "not_recom"], ordered=True)
df.target = df.target.astype("category", categories=targets, ordered=True)

print (df)
fig, ax = plt.subplots(figsize=(11, 9))
df1 = pd.get_dummies(df)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df1.corr(), ax=ax, cmap=cmap)
