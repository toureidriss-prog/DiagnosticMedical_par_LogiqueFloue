import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


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
# LOAD RULES
# =========================
rules_df = pd.read_csv("rules.csv")

# =========================
# FUZZY SETUP
# =========================
thalach = ctrl.Antecedent(np.arange(60, 221, 1), 'thalach')
oldpeak = ctrl.Antecedent(np.arange(0, 7.1, 0.1), 'oldpeak')
ca = ctrl.Antecedent(np.arange(0, 4, 1), 'ca')
thal = ctrl.Antecedent(np.array([3,6,7]), 'thal')
num = ctrl.Consequent(np.arange(0, 4.1, 0.1), 'num')

# (tu peux copier exactement tes MF ici — inchangé)
# ...

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
# N = len(data)

# =========================
# BUILD SYSTEM
# =========================
fuzzy_rules = [
    ctrl.Rule(
        thalach[r.thalach] &
        oldpeak[r.oldpeak] &
        ca[r.ca] &
        thal[r.thal],
        num[r.num]
    )
    for _, r in rules_df.iterrows()
]

system = ctrl.ControlSystem(fuzzy_rules)

# IMPORTANT: new simulation per prediction
def predict(thalach_val, oldpeak_val, ca_val, thal_val):

    sim = ctrl.ControlSystemSimulation(system)

    sim.input['thalach'] = thalach_val
    sim.input['oldpeak'] = oldpeak_val
    sim.input['ca'] = ca_val
    sim.input['thal'] = thal_val

    sim.compute()

    x = sim.output['num']

    if x < 1.5:
        return 0, x
    elif x < 2.2:
        return 1, x
    elif x < 2.8:
        return 2, x
    elif x < 3.5:
        return 3, x
    else:
        return 4, x


# =========================
# EXPLANATION (IMPORTANT)
# =========================
def explain_rule(thalach_val, oldpeak_val, ca_val, thal_val):

    best_rule = None
    best_activation = 0

    for _, r in rules_df.iterrows():

        activation = min(
            thalach[r.thalach].mf[np.argmin(np.abs(thalach.universe - thalach_val))],
            oldpeak[r.oldpeak].mf[np.argmin(np.abs(oldpeak.universe - oldpeak_val))],
            ca[r.ca].mf[np.argmin(np.abs(ca.universe - ca_val))],
            thal[r.thal].mf[np.argmin(np.abs(thal.universe - thal_val))]
        )

        if activation > best_activation:
            best_activation = activation
            best_rule = r

    rule_text = f"""
SI thalach = {best_rule.thalach}
ET oldpeak = {best_rule.oldpeak}
ET ca = {best_rule.ca}
ET thal = {best_rule.thal}
ALORS num = {best_rule.num}
"""

    return rule_text, best_activation


# =========================
# METRICS (OPTIONAL)
# =========================
def get_rules_count():
    return len(rules_df)