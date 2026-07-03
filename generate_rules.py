import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import itertools
from ucimlrepo import fetch_ucirepo

# =========================
# DATASET
# =========================
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
df = pd.concat([X, y], axis=1)
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

data = df[['thalach', 'oldpeak', 'ca', 'thal', 'num']]

# =========================
# TERMES
# =========================
thalach_terms = ['faible','moyenne','elevee']
oldpeak_terms = ['leger','risque','severe']
ca_terms = ['zero','un','deux','trois']
thal_terms = ['normal','fixe','reversible']
num_terms = ['sain','faible','modere','severe','tres_severe']

# =========================
# MEMBERSHIP (IDENTIQUE À TON CODE)
# =========================
thalach = ctrl.Antecedent(np.arange(60, 221, 1), 'thalach')
oldpeak = ctrl.Antecedent(np.arange(0, 7.1, 0.1), 'oldpeak')
ca = ctrl.Antecedent(np.arange(0, 4, 1), 'ca')
thal = ctrl.Antecedent(np.array([3,6,7]), 'thal')
num = ctrl.Consequent(np.arange(0, 4.1, 0.1), 'num')

thalach['faible'] = fuzz.trapmf(thalach.universe,[60,60,100,141])
thalach['moyenne'] = fuzz.trimf(thalach.universe,[111,152,194])
thalach['elevee'] = fuzz.trapmf(thalach.universe,[152,216,220,220])

oldpeak['leger'] = fuzz.trapmf(oldpeak.universe,[0,0,1,2])
oldpeak['risque'] = fuzz.trimf(oldpeak.universe,[1.5,2.8,4.2])
oldpeak['severe'] = fuzz.trapmf(oldpeak.universe,[2.55,4,7,7])

ca['zero'] = fuzz.trimf(ca.universe,[0,0,0])
ca['un'] = fuzz.trimf(ca.universe,[1,1,1])
ca['deux'] = fuzz.trimf(ca.universe,[2,2,2])
ca['trois'] = fuzz.trimf(ca.universe,[3,3,3])

thal['normal'] = fuzz.trimf(thal.universe,[3,3,3])
thal['fixe'] = fuzz.trimf(thal.universe,[6,6,6])
thal['reversible'] = fuzz.trimf(thal.universe,[7,7,7])

num['sain'] = fuzz.trapmf(num.universe,[0,0,0.3,1])
num['faible'] = fuzz.trimf(num.universe,[0,1,2])
num['modere'] = fuzz.trimf(num.universe,[1,2,3])
num['severe'] = fuzz.trimf(num.universe,[2,3,4])
num['tres_severe'] = fuzz.trapmf(num.universe,[3,3.7,4,4])

# =========================
# MEMBERSHIP FUNCTION
# =========================
def mfs(row):
    return {
        ('thalach','faible'): fuzz.interp_membership(thalach.universe, thalach['faible'].mf, row['thalach']),
        ('thalach','moyenne'): fuzz.interp_membership(thalach.universe, thalach['moyenne'].mf, row['thalach']),
        ('thalach','elevee'): fuzz.interp_membership(thalach.universe, thalach['elevee'].mf, row['thalach']),

        ('oldpeak','leger'): fuzz.interp_membership(oldpeak.universe, oldpeak['leger'].mf, row['oldpeak']),
        ('oldpeak','risque'): fuzz.interp_membership(oldpeak.universe, oldpeak['risque'].mf, row['oldpeak']),
        ('oldpeak','severe'): fuzz.interp_membership(oldpeak.universe, oldpeak['severe'].mf, row['oldpeak']),

        ('ca','zero'): fuzz.interp_membership(ca.universe, ca['zero'].mf, row['ca']),
        ('ca','un'): fuzz.interp_membership(ca.universe, ca['un'].mf, row['ca']),
        ('ca','deux'): fuzz.interp_membership(ca.universe, ca['deux'].mf, row['ca']),
        ('ca','trois'): fuzz.interp_membership(ca.universe, ca['trois'].mf, row['ca']),

        ('thal','normal'): fuzz.interp_membership(thal.universe, thal['normal'].mf, row['thal']),
        ('thal','fixe'): fuzz.interp_membership(thal.universe, thal['fixe'].mf, row['thal']),
        ('thal','reversible'): fuzz.interp_membership(thal.universe, thal['reversible'].mf, row['thal']),

        ('num','sain'): fuzz.interp_membership(num.universe, num['sain'].mf, row['num']),
        ('num','faible'): fuzz.interp_membership(num.universe, num['faible'].mf, row['num']),
        ('num','modere'): fuzz.interp_membership(num.universe, num['modere'].mf, row['num']),
        ('num','severe'): fuzz.interp_membership(num.universe, num['severe'].mf, row['num']),
        ('num','tres_severe'): fuzz.interp_membership(num.universe, num['tres_severe'].mf, row['num']),
    }

memberships = [mfs(row) for _, row in data.iterrows()]
N = len(data)

# =========================
# RULES GENERATION
# =========================
rules = list(itertools.product(
    thalach_terms,
    oldpeak_terms,
    ca_terms,
    thal_terms,
    num_terms
))

results = []

for r in rules:
    thalach_t, oldpeak_t, ca_t, thal_t, num_t = r

    sup_ab = sup_a = sup_b = 0

    for p in memberships:

        A = min(
            p[('thalach', thalach_t)],
            p[('oldpeak', oldpeak_t)],
            p[('ca', ca_t)],
            p[('thal', thal_t)]
        )

        B = p[('num', num_t)]

        sup_a += A
        sup_b += B
        sup_ab += min(A, B)

    support = sup_ab / N
    confidence = sup_ab / sup_a if sup_a > 0 else 0
    lift = confidence / (sup_b / N) if sup_b > 0 else 0

    results.append([thalach_t, oldpeak_t, ca_t, thal_t, num_t, support, confidence, lift])

rules_df = pd.DataFrame(results, columns=[
    "thalach","oldpeak","ca","thal","num","support","confidence","lift"
])

selected = rules_df[
    (rules_df["support"] >= 0.003) &
    (rules_df["confidence"] >= 0.6) &
    (rules_df["lift"] > 1)
]

selected.to_csv("rules.csv", index=False)

print("Règles sauvegardées :", len(selected))