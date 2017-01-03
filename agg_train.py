import datetime
import sys
import numpy as np
import pandas as pd
from group_to_features import group_to_features

np.random.seed(1337)

prefix = sys.argv[1]
target = sys.argv[2]

d = pd.read_csv(prefix + target + '.csv',
        parse_dates=[
            'measurement_date',
            'first_contract_date'])
dg = d.groupby(['customer_id'])
l = len(dg) / 1000

positive = []
negative = []

j = 1
for k, g in dg:
    if j % 1000 == 0:
        print j / 1000, '/', l
    j += 1
    if len(g[target].dropna().value_counts()) > 1 and g[target].dropna().tolist()[0] == 0.0:
        i = len(g)
        while (g[target].head(i).tail(2).tolist() != [0.0, 1.0]) and i > 0:
            i -= 1
        positive.append(group_to_features(g.head(i), 1))
    else:
        if len(g) > 1:
            i = np.random.randint(1, len(g) + 1)
        else:
            i = len(g)
        negative.append(group_to_features(g.head(i), 0))

d = pd.DataFrame(positive + negative)
d = d.iloc[np.random.permutation(len(d))]
d.to_csv('agg' + prefix[3:] + target + '.csv', index=False)
