import datetime
import sys
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.decomposition

np.random.seed(1337)

prefix = sys.argv[1]
components = int(sys.argv[2])

products = '''ind_ahor_fin_ult1
ind_aval_fin_ult1
ind_cco_fin_ult1
ind_cder_fin_ult1
ind_cno_fin_ult1
ind_ctju_fin_ult1
ind_ctma_fin_ult1
ind_ctop_fin_ult1
ind_ctpp_fin_ult1
ind_deco_fin_ult1
ind_deme_fin_ult1
ind_dela_fin_ult1
ind_ecue_fin_ult1
ind_fond_fin_ult1
ind_hip_fin_ult1
ind_plan_fin_ult1
ind_pres_fin_ult1
ind_reca_fin_ult1
ind_tjcr_fin_ult1
ind_valo_fin_ult1
ind_viv_fin_ult1
ind_nomina_ult1
ind_nom_pens_ult1
ind_recibo_ult1'''.split('\n')

d = pd.read_csv(prefix + 'part0.csv')
d['target_name'] = 'test'
for i in range(1, 10):
    print 'reading', i
    e = pd.read_csv(prefix + 'part{0}.csv'.format(i))
    e['target_name'] = 'test'
    d = pd.concat([d, e])

for p in products:
    print 'reading', p
    e = pd.read_csv(prefix + 'train_{0}.csv'.format(p))
    e['target_name'] = p
    d = pd.concat([d, e])

cols_to_encode = []
for c in d.columns:
    if d[c].dtype == object and c not in ['customer_id', 'target_name', 'target']:
        cols_to_encode.append(c)
d = pd.get_dummies(d, columns=cols_to_encode)

d.index = pd.Index(range(len(d)))

cids = d.customer_id.copy()
targets = d.target.copy()
target_names = d.target_name.copy()

del d['customer_id']
del d['target']
del d['target_name']

print 'deleting unnecessary columns'
for c in d.columns.tolist():
    if c.endswith('_no'):
        del d[c]

d.fillna(0, inplace=True)

#print 'normalizing'
#d = sklearn.preprocessing.normalize(d, axis=0)

print 'pca'
pca = sklearn.decomposition.PCA(n_components=components, copy=False)
d = pca.fit_transform(d)
gc.collect()

d = pd.DataFrame(d, index=pd.Index(range(len(d))))

d.columns = ['pc_{0}'.format(i) for i in range(components)]

d['customer_id'] = cids
d['target'] = targets
d['target_name'] = target_names

d[d.target_name == 'test'].to_csv('pc{0}'.format(components) + prefix[3:] + 'test.csv', index=False)
for p in products:
    print 'writing', p
    d[d.target_name == p].to_csv('pc{0}'.format(components) + prefix[3:] + 'train_{0}.csv'.format(p), index=False)
