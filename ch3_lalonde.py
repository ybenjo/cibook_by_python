# wget https://users.nber.org/~rdehejia/data/cps_controls.dta
# wget https://users.nber.org/~rdehejia/data/cps_controls3.dta
# wget https://users.nber.org/~rdehejia/data/nsw_dw.dta
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import lightgbm as lgbm

nsw = pd.read_stata('./data/nsw_dw.dta')
cps1 = pd.read_stata('./data/cps_controls.dta')
cps3 = pd.read_stata('./data/cps_controls3.dta')

cps1_nsw = pd.concat([cps1, nsw[nsw.treat == 1]])
cps1_nsw = cps1_nsw.reset_index(drop=True)
cps3_nsw = pd.concat([cps3, nsw[nsw.treat == 1]])

# 3.4.2
model = LinearRegression()
model.fit(nsw[['treat', 're74', 're75', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married']], nsw.re78)
print(f'RCT treatment:{model.coef_[0]}')

# result
# RCT treatment:1676.3425

# 3.4.3
model.fit(cps1_nsw[['treat', 're74', 're75', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married']], cps1_nsw.re78)
print(f'cps1_nsw treatment:{model.coef_[0]}')
# result
# cps1_nsw treatment:699.1405029296875

model.fit(cps3_nsw[['treat', 're74', 're75', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married']], cps3_nsw.re78)
print(f'cps3_nsw treatment:{model.coef_[0]}')
# result
# cps3_nsw treatment:1548.2431640625

# 3.4.4
# calc propensity score
ps_model = LogisticRegression()
# 本来の R のコードではこの追加した2列を加えて回帰しているが，
# 自分が試したところ treatment が -200 になるという感じで propensity score の品質が低下した
# この人のコードでも二乗項を無視している? https://github.com/nekoumei/cibook-python/blob/master/notebook/ch3_lalonde.ipynb
cps1_nsw['re74_2'] = cps1_nsw['re74'] ** 2
cps1_nsw['re75_2'] = cps1_nsw['re75'] ** 2
ps_model.fit(cps1_nsw[['re74', 're75', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married']], cps1_nsw.treat)
cps1_nsw['ps'] = ps_model.predict_proba(cps1_nsw[['re74', 're75', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married']])[:, 1]

# greedy-matching
tuples = list(zip(cps1_nsw.index.tolist(), cps1_nsw.ps.tolist(), cps1_nsw.treat.tolist()))
treatment_data = [tpl for tpl in tuples if tpl[-1] == 1]
controll_data = [tpl for tpl in tuples if tpl[-1] == 0]

n_match = min(len(treatment_data), len(controll_data))
treatment_data = sorted(treatment_data, key=lambda x: x[1])

matched_indices = []
for i in tqdm(range(n_match), ascii=True):
    idx_i = treatment_data[i][0]
    ps_i = treatment_data[i][1]
    j, diff = sorted([(j, (ps_i - e[1]) ** 2) for j, e in enumerate(controll_data)], key=lambda x: x[1])[0]
    idx_j = controll_data.pop(j)[0]
    matched_indices.append(idx_i)
    matched_indices.append(idx_j)
    
matched_data = cps1_nsw.iloc[matched_indices]
model.fit(matched_data[['treat']], matched_data.re78)
print(f'After matching treatment:{model.coef_[0]}')

# result
# After matching treatment:1419.0723876953125
# 少しはまともになった


# バランスを考える
def calc_balance(orig_data, matched_data, key):
    x_1_unadj = orig_data[orig_data.treat == 1][key].tolist()
    x_0_unadj = orig_data[orig_data.treat == 0][key].tolist()
    s_c_unadj = len(x_1_unadj) * (np.std(x_1_unadj) ** 2) + len(x_0_unadj) * (np.std(x_0_unadj) ** 2)
    s_c_unadj /= (len(x_1_unadj) + len(x_0_unadj))
    d_unadj = abs(np.average(x_1_unadj) - np.average(x_0_unadj)) / (s_c_unadj ** 0.5)

    x_1_adj = matched_data[matched_data.treat == 1][key].tolist()
    x_0_adj = matched_data[matched_data.treat == 0][key].tolist()
    s_c_adj = len(x_1_adj) * (np.std(x_1_adj) ** 2) + len(x_0_adj) * (np.std(x_0_adj) ** 2)
    s_c_adj /= (len(x_1_adj) + len(x_0_adj))
    d_adj = abs(np.average(x_1_adj) - np.average(x_0_adj)) / (s_c_adj ** 0.5)

    return (d_unadj, d_adj)


for k in ['ps', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married', 're74', 're75']:
    d_unadj, d_adj = calc_balance(cps1_nsw, matched_data, k)
    print(f'{k}:{d_unadj} -> {d_adj}')
    
# result
# ps:5.103438518915803 -> 0.0008180420230336148
# age:0.6730570008914505 -> 0.05865811234871624
# education:0.5874806622040686 -> 0.06644815719721366
# black:2.9331646790098183 -> 0.07195175669120617
# hispanic:0.04868863047775604 -> 0.22810637940488046
# nodegree:0.9033203143027307 -> 0.20640627484613447
# married:1.1552913234585143 -> 0.013878184541334224
# re74:1.251062287666147 -> 0.0612110927629351
# re75:1.313920874084624 -> 0.02003358291795212

# IPW による効果推定
cps1_nsw['weight'] = 1 / ps_model.predict_proba(cps1_nsw[['re74', 're75', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married']])[:, 1]
model.fit(cps1_nsw[['treat']], cps1_nsw.re78, sample_weight=cps1_nsw['weight'])
print(f'IPW treatment:{model.coef_[0]}')
# result
# IPW treatment:-14884.73046875

# 3.4.4 の発展
# propensity score を lightgbm で求めたらどうなるか試す
ps_model = lgbm.LGBMClassifier(n_estimators=10)
# 本来の R のコードではこの追加した2列を加えて回帰しているが，
# 自分が試したところ treatment が -200 になるという感じで propensity score の品質が低下した
# この人のコードでも二乗項を無視している? https://github.com/nekoumei/cibook-python/blob/master/notebook/ch3_lalonde.ipynb
cps1_nsw['re74_2'] = cps1_nsw['re74'] ** 2
cps1_nsw['re75_2'] = cps1_nsw['re75'] ** 2
ps_model.fit(cps1_nsw[['re74', 're75', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married']], cps1_nsw.treat)
cps1_nsw['ps'] = ps_model.predict_proba(cps1_nsw[['re74', 're75', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married']])[:, 1]

# greedy-matching
tuples = list(zip(cps1_nsw.index.tolist(), cps1_nsw.ps.tolist(), cps1_nsw.treat.tolist()))
treatment_data = [tpl for tpl in tuples if tpl[-1] == 1]
controll_data = [tpl for tpl in tuples if tpl[-1] == 0]

n_match = min(len(treatment_data), len(controll_data))
treatment_data = sorted(treatment_data, key=lambda x: x[1])

matched_indices = []
for i in tqdm(range(n_match), ascii=True):
    idx_i = treatment_data[i][0]
    ps_i = treatment_data[i][1]
    j, diff = sorted([(j, (ps_i - e[1]) ** 2) for j, e in enumerate(controll_data)], key=lambda x: x[1])[0]
    idx_j = controll_data.pop(j)[0]
    matched_indices.append(idx_i)
    matched_indices.append(idx_j)
    
matched_data = cps1_nsw.iloc[matched_indices]
model.fit(matched_data[['treat']], matched_data.re78)
print(f'After matching treatment w/ lightgbm:{model.coef_[0]}')

# result
# After matching treatment w/ lightgbm:1657.2574462890625
# RCT の結果に近付きましたね

# バランスを考える
for k in ['ps', 'age', 'education', 'black', 'hispanic', 'nodegree', 'married', 're74', 're75']:
    d_unadj, d_adj = calc_balance(cps1_nsw, matched_data, k)
    print(f'{k}:{d_unadj} -> {d_adj}')

# result
# ps:10.099871800781095 -> 0.845215828966668
# age:0.6730570008914505 -> 0.0972967209844437
# education:0.5874806622040686 -> 0.07424385005078368
# black:2.9331646790098183 -> 0.04371301894719376
# hispanic:0.04868863047775604 -> 0.07352146220938079
# nodegree:0.9033203143027307 -> 0.19520896517421565
# married:1.1552913234585143 -> 0.19364916731037085
# re74:1.251062287666147 -> 0.012639302570196803
# re75:1.313920874084624 -> 0.045295152616059715

# 結論
# 一見改善したように見えるけれど n_estimators によっては過学習するし共変量の cohen d が大きくなっている?
