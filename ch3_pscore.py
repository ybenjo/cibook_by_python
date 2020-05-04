# wget http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import networkx as nx
from networkx.algorithms.bipartite.matching import hopcroft_karp_matching
from tqdm import tqdm
from load_male_data import load_biased_male_data


biased_male_data = load_biased_male_data()

# 3.2.1
# まずは傾向スコアを求める
model = LogisticRegression()
model.fit([[e['history'], e['recency']] + e['binarized_channel'] for e in biased_male_data],
          [e['treatment'] for e in biased_male_data])

propensity_scores = model.predict_proba([[e['history'], e['recency']] + e['binarized_channel'] for e in biased_male_data])[:, 1]

# どうせだし biased_male_data に propensity_scores を付与しましょう
for pos, e in enumerate(biased_male_data):
    e['ps'] = propensity_scores[pos]

# 群に分割
treatment_data = [e for e in biased_male_data if e['treatment'] == 1]
controll_data = [e for e in biased_male_data if e['treatment'] == 0]

# # 最大重みマッチングとして解きましょう
# I = range(len(treatment_data))
# J = range(len(controll_data))
# g = nx.Graph()
# weighted_edges = []
# for i in tqdm(I, ascii=True):
#     for j in J:
#         # 通常は diff^2 が距離なのでそのまま最小で使いたいですが
#         # 最大で解くので 1.0 から引きます
#         w = 1.0 - (treatment_data[i]['ps'] - controll_data[j]['ps']) ** 2
#         weighted_edges.append((f'i_{i}', f'j_{j}', w))

# g.add_weighted_edges_from(weighted_edges)

# ret = hopcroft_karp_matching(g)
# # マッチング結果を matched_data に入れる
# matched_data = []
# for k, v in ret.items():
#     _type, v1 = k.split('_')
#     v1 = int(v1)
#     v2 = int(v.split('_')[1])
#     if _type == 'i':
#         matched_data.append(treatment_data[v1])
#         matched_data.append(controll_data[v2])
#     else:
#         matched_data.append(treatment_data[v2])
#         matched_data.append(controll_data[v1])

# # treatment:0.843491048593346
# # recency:0.4778351631443985 -> 0.4801651891393515
# # history:0.2716497558095329 -> 0.2719059723736311
# # 全然改善しませんね

# greedy に解きましょう
# propensity score で昇順にソート ( treatment < controll だから ps が大きいものは余らせたい)
treatment_data = sorted(treatment_data, key=lambda x: x['ps'])
n_match = min(len(treatment_data), len(controll_data))
matched_data = []
for i in tqdm(range(n_match), ascii=True):
    e_pos = treatment_data[i]
    # 一番近いものを探す
    j = sorted([(j, (e['ps'] - e_pos['ps']) ** 2) for j, e in enumerate(controll_data)], key=lambda x: x[1])[0][0]
    e_neg = controll_data.pop(j)
    matched_data.append(e_pos)
    matched_data.append(e_neg)

# マッチが終わったので効果を計算する
model = LinearRegression()
model.fit([[e['treatment']]  for e in matched_data],
          [e['spend'] for e in matched_data])
print(f'treatment:{model.coef_[0]}')
# result
# treatment:0.8451103782474086

# 3.2.3
# マッチング前後で共変量の分布がどう変わったか確認しましょう
# 各共変量に対して Average Standardized Absolute Mean distance (ASAM) を計算しましょう
# ASAM は平均の差をその標準誤差で割った値です

# 本文の「その標準偏差」の「その」が何を指すのかわからない
# 検索したところ， Cohen の d がこれに相当するのではないか?
# 定義は |E[x_1] - E[x_2]| / s_c
# s_c = ((n_1 * (s_1 ** 2) + n_2 * (s_2 ** 2)) / (n_1 + n_2)) ** 0.5
# s_1 = Var[x_1] ** 0.5, s_2 = Var[x_2] ** 0.5 
# 参考 : https://bellcurve.jp/statistics/course/12765.html
# figure 3.5 の `distance` が何を意味するのかよく分かっていない
# ひとまず ps で計算してみましょう

x_1_unadj = [x['ps'] for x in biased_male_data if x['treatment'] == 1]
x_0_unadj = [x['ps'] for x in biased_male_data if x['treatment'] == 0]
s_c_unadj = len(x_1_unadj) * (np.std(x_1_unadj) ** 2) + len(x_0_unadj) * (np.std(x_0_unadj) ** 2)
s_c_unadj /= (len(x_1_unadj) + len(x_0_unadj))
d_unadj = abs(np.average(x_1_unadj) - np.average(x_0_unadj)) / (s_c_unadj ** 0.5)

x_1_adj = [x['ps'] for x in matched_data if x['treatment'] == 1]
x_0_adj = [x['ps'] for x in matched_data if x['treatment'] == 0]
s_c_adj = len(x_1_adj) * (np.std(x_1_adj) ** 2) + len(x_0_adj) * (np.std(x_0_adj) ** 2)
s_c_adj /= (len(x_1_adj) + len(x_0_adj))
d_adj = abs(np.average(x_1_adj) - np.average(x_0_adj)) / (s_c_adj ** 0.5)
print(f'ps:{d_unadj} -> {d_adj}')

for k in ['recency', 'history']:
    x_1_unadj = [x[k] for x in biased_male_data if x['treatment'] == 1]
    x_0_unadj = [x[k] for x in biased_male_data if x['treatment'] == 0]
    s_c_unadj = len(x_1_unadj) * (np.std(x_1_unadj) ** 2) + len(x_0_unadj) * (np.std(x_0_unadj) ** 2)
    s_c_unadj /= (len(x_1_unadj) + len(x_0_unadj))
    d_unadj = abs(np.average(x_1_unadj) - np.average(x_0_unadj)) / (s_c_unadj ** 0.5)

    x_1_adj = [x[k] for x in matched_data if x['treatment'] == 1]
    x_0_adj = [x[k] for x in matched_data if x['treatment'] == 0]
    s_c_adj = len(x_1_adj) * (np.std(x_1_adj) ** 2) + len(x_0_adj) * (np.std(x_0_adj) ** 2)
    s_c_adj /= (len(x_1_adj) + len(x_0_adj))
    d_adj = abs(np.average(x_1_adj) - np.average(x_0_adj)) / (s_c_adj ** 0.5)

    print(f'{k}:{d_unadj} -> {d_adj}')

for pos in range(3):
    x_1_unadj = [x['binarized_channel'][pos] for x in biased_male_data if x['treatment'] == 1]
    x_0_unadj = [x['binarized_channel'][pos] for x in biased_male_data if x['treatment'] == 0]
    s_c_unadj = len(x_1_unadj) * (np.std(x_1_unadj) ** 2) + len(x_0_unadj) * (np.std(x_0_unadj) ** 2)
    s_c_unadj /= (len(x_1_unadj) + len(x_0_unadj))
    d_unadj = abs(np.average(x_1_unadj) - np.average(x_0_unadj)) / (s_c_unadj ** 0.5)

    x_1_adj = [x['binarized_channel'][pos] for x in matched_data if x['treatment'] == 1]
    x_0_adj = [x['binarized_channel'][pos] for x in matched_data if x['treatment'] == 0]
    s_c_adj = len(x_1_adj) * (np.std(x_1_adj) ** 2) + len(x_0_adj) * (np.std(x_0_adj) ** 2)
    s_c_adj /= (len(x_1_adj) + len(x_0_adj))
    d_adj = abs(np.average(x_1_adj) - np.average(x_0_adj)) / (s_c_adj ** 0.5)

    print(f'channel_{pos}:{d_unadj} -> {d_adj}')
    
# result
# ps:0.5235731806205225 -> 0.3186743724862133
# recency:0.4778351631443985 -> 0.3321028493729235
# history:0.2716497558095329 -> 0.01041296968179367
# channel_0:0.18216784523404742 -> 0.007067847374114813
# channel_1:0.07359552251750925 -> 0.004461431671379047
# channel_2:0.04637906531038619 -> 0.008512051815361875
