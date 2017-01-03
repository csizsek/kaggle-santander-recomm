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

cids = d.customer_id.copy()
targets = d.target.copy()
target_names = d.target_name.copy()

del d['customer_id']
del d['target']
del d['target_name']

print 'dummyfying'
cols_to_encode = []
for c in d.columns:
    if d[c].dtype == object:
        cols_to_encode.append(c)
d = pd.get_dummies(d, columns=cols_to_encode)

print 'deleting unnecessary columns'
for c in d.columns.tolist():
    if c.endswith('_no'):
        del d[c]

d.fillna(0, inplace=True)

print 'pca'
pca = sklearn.decomposition.PCA(n_components=components)
pca = pca.fit(d)
e = pca.transform(d)
del d
gc.collect()

print 'constructing data frame'
f = pd.DataFrame(e, index=pd.Index(range(len(e))))
del e
gc.collect()

f.columns = ['pc_{0}'.format(i) for i in range(components)]

f['customer_id'] = cids
f['target'] = targets
f['target_name'] = target_names

print 'writing all parts'
f.to_csv('pc{0}'.format(components) + prefix[3:] + 'test.csv', index=False)
del f
gc.collect()


for p in products:
    print 'reading', p
    d = pd.read_csv(prefix + 'train_{0}.csv'.format(p))
    e['target_name'] = p

    cids = d.customer_id.copy()
    targets = d.target.copy()
    target_names = d.target_name.copy()

    del d['customer_id']
    del d['target']
    del d['target_name']

    cols_to_encode = []
    for c in d.columns:
        if d[c].dtype == object:
            cols_to_encode.append(c)
    d = pd.get_dummies(d, columns=cols_to_encode)

    print 'deleting unnecessary columns'
    for c in d.columns.tolist():
        if c.endswith('_no'):
            del d[c]

    d.fillna(0, inplace=True)

    print 'pca'
    e = pca.transform(d)
    del d
    gc.collect()

    print 'constructing data frame'
    f = pd.DataFrame(e, index=pd.Index(range(len(e))))
    del e
    gc.collect()

    f.columns = ['pc_{0}'.format(i) for i in range(components)]

    f['customer_id'] = cids
    f['target'] = targets
    f['target_name'] = target_names

    f.to_csv('pc{0}'.format(components) + prefix[3:] + 'train_{0}.csv'.format(p), index=False)
    del f
    gc.collect()
