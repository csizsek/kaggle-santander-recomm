import datetime
import sys
import numpy as np
import pandas as pd
from group_to_features import group_to_features

np.random.seed(1337)

input_file = sys.argv[1]

d = pd.read_csv(input_file,
        parse_dates=[
            'measurement_date',
            'first_contract_date'])
dg = d.groupby(['customer_id'])
l = len(dg) / 1000

instances = []

j = 1
for k, g in dg:
    if j % 1000 == 0:
        print j / 1000, '/', l
    j += 1
    instances.append(group_to_features(g, np.nan))

d = pd.DataFrame(instances)
d = d.iloc[np.random.permutation(len(d))]
d.to_csv('agg' + input_file[3:], index=False)
