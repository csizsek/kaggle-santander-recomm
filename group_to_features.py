import datetime
import numpy as np
import pandas as pd

np.random.seed(1337)

def group_to_features(g, target):
    g = g.copy().sort_values('measurement_date')

    d = {}

    d['customer_id'] = g.customer_id.max()
    d['target'] = target

    last_id = g.index.tolist()[-1]
    for p in g.columns[23:]:
        g.set_value(last_id, p, np.nan)

    try:
        d['measurement_month_last'] = g.measurement_date.max().month
    except:
        d['measurement_month_last'] = np.nan

    d['gender'] = g.gender.max()
    d['age'] = g.age.max()

    try:
        d['first_contract_month'] = g.first_contract_date.max().month
    except:
        d['first_contract_month'] = np.nan

    try:
        d['age_at_first_contract'] = g.age.max() - (2016 - g.first_contract_date.max().year)
    except:
        d['age_at_first_contract'] = np.nan

    try:
        d['diff_first_contract_month_and_last_month'] = (g.first_contract_date.max().month - g.measurement_date.max().month) % 12
    except:
        d['diff_first_contract_month_and_last_month'] = np.nan

    try:
        d['first_contract_months_ago'] = (g.measurement_date.max() - g.first_contract_date.max()).days / 30
    except:
        d['first_contract_months_ago'] = np.nan

    d['customer_for_months'] = g.customer_for_months.max()

    try:
        d['diff_first_customer_month_and_last_month'] = (g.measurement_date.max().month - g.customer_for_months.max()) % 12
    except:
        d['diff_first_customer_month_and_last_month'] = np.nan

    try:
        d['diff_first_customer_month_and_first_contract_month'] = (g.first_contract_date.max().month - g.customer_for_months.max()) % 12
    except:
        d['diff_first_customer_month_and_first_contract_month'] = np.nan

    d['is_foreigner'] = g.is_foreigner.max()
    d['join_channel'] = g.join_channel.max()
    d['is_dead'] = np.nan
    if len(g.is_dead.dropna().tolist()) > 0:
        d['is_dead'] = g.is_dead.dropna().tolist()[-1]

    try:
        d['household_income_diff'] = g['household_income'].tolist()[-1] - g['household_income'].tolist()[0]
    except:
        d['household_income_diff'] = np.nan

    lca_features = [
        'employment_index',
        'country_of_residence',
        'is_new_customer',
        'is_primary_customer',
        'customer_type',
        'customer_relation_type',
        'country_residence_bank_same',
        'is_spouse_of_an_employee',
        'province_code',
        'is_active_customer',
        'household_income',
        'customer_segment']

    for p in g.columns[23:]:
        lca_features.append(p)
        d[p + '_sum'] = g[p].sum()

    for f in lca_features:
        try:
            d[f + '_last'] = g[f].dropna().tolist()[-1]
        except:
            d[f + '_last'] = np.nan

        d[f + '_changed'] = 'yes' if len(g[f].dropna().value_counts()) > 1 else 'no'
        d[f + '_changed_measurements_ago'] = np.nan
        if d[f + '_changed'] == 'yes':
            i = 2
            while len(g[f].dropna().tail(i).value_counts()) == 1:
                i += 1
            d[f + '_changed_measurements_ago'] = i - 1

    return d
