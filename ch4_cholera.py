from math import log
from collections import Counter
import pandas as pd
from sklearn.linear_model import LinearRegression

sv_1849 = [283, 157, 192, 249, 259, 226, 352, 97, 111, 8, 235, 92]
lsv_1849 = [256, 267, 312, 257, 318, 446, 143, 193, 243, 215, 544, 187, 153, 81, 113, 176]
sv_1854 = [371, 161, 148, 362, 244, 237, 282, 59, 171, 9, 240, 174]
lsv_1854 = [113, 174, 270, 93, 210, 388, 92, 58, 117, 49, 193, 303, 142, 48, 165, 132]

sv_death = sv_1849 + sv_1854
lsv_death = lsv_1849 + lsv_1854

sv_area = [f'sv_{x + 1}' for x in range(len(sv_1849))] + [f'sv_{x + 1}' for x in range(len(sv_1854))]
lsv_area = [f'lsv_{x + 1}' for x in range(len(lsv_1849))] + [f'lsv_{x + 1}' for x in range(len(lsv_1854))]
sv_year = [1849 for x in range(len(sv_1849))] + [1854 for x in range(len(sv_1854))]
lsv_year = [1849 for x in range(len(lsv_1849))] + [1854 for x in range(len(lsv_1854))]

sv = pd.DataFrame({
    'area': sv_area,
    'year': sv_year,
    'death': sv_death,
    'LSV': [0] * len(sv_area),
    'company': ['Southwark and Vauxhall'] * len(sv_area)
})

lsv = pd.DataFrame({
    'area': lsv_area,
    'year': lsv_year,
    'death': lsv_death,
    'LSV': [1] * len(lsv_area),
    'company': ['Lambeth & Southwark and Vauxhall'] * len(lsv_area)
})

js_df = pd.concat([sv, lsv])
# この操作どういう意味があるのだろう
js_df.LSV = js_df.company.apply(lambda x: 1 if x == 'Lambeth & Southwark and Vauxhall' else 0)

# 面倒になってしまったので pandas を捨てます
js_sum = Counter()
for _, e in js_df.iterrows():
    js_sum[(e.company, e.LSV, e.year)] += e.death

# 集計による推定
# pandas 面倒くさくなったので素直に引き算する
lsv_1849_death = js_sum[('Lambeth & Southwark and Vauxhall', 1, 1849)]
lsv_1854_death = js_sum[('Lambeth & Southwark and Vauxhall', 1, 1854)]
lsv_gap = lsv_1854_death - lsv_1849_death
lsv_gap_rate = lsv_1854_death / lsv_1849_death - 1
print(f"Lambeth & Southwark and Vauxhall:{lsv_gap},{lsv_gap_rate}")

sv_1849_death = js_sum[('Southwark and Vauxhall', 0, 1849)]
sv_1854_death = js_sum[('Southwark and Vauxhall', 0, 1854)]
sv_gap = sv_1854_death - sv_1849_death
sv_gap_rate = sv_1854_death / sv_1849_death - 1
print(f"Southwark and Vauxhall:{sv_gap},{sv_gap_rate}")

# sv and lsv effect
total_1849_death = sv_1849_death + lsv_1849_death
total_1854_death = sv_1854_death + lsv_1854_death
total_gap = lsv_gap - sv_gap
# 本文 : 「よって，この分析の結果による効果量はそれらの差分をとって-1,554若しくは-43%ということになり」の -43% の分母は何かが分からない
print(f'total:{total_gap}')

# 差の差を考えましょう
# 平行トレンド仮定より，sv_gap 分だけ本来は増加していたはずなので sv_gap 分も減少していると考える
did_gap = lsv_gap - sv_gap
did_gap_rate = (lsv_1849_death + did_gap) / lsv_1849_death - 1
print(f'did:{did_gap},{did_gap_rate}')

# result
# Lambeth & Southwark and Vauxhall:-1357,-0.3475922131147541
# Southwark and Vauxhall:197,0.08712958867757625
# total:-1554,-0.31048951048951046
# did:-1554,-0.3980532786885246

# 4.1.4
# 回帰で DID を計算する
X = []
y = []

for k, v in js_sum.items():
    fv = []
    # js_sum の key は (company, lsv, year) なので
    _, lsv, year = k
    d1854 = 1 if year == 1854 else 0
    y.append(v)
    X.append([lsv * d1854, lsv, d1854])

model = LinearRegression()
model.fit(X, y)
print(f'treatment:{model.coef_[0]}')

# result
# treatment:-1554.0

# エリア別に回帰する
# js_df に D1854 と D1854xLSV を追加する
js_df['D1854'] = js_df.year.apply(lambda x: 1 if x == 1854 else 0)
js_df['D1854xLSV'] = js_df.D1854 * js_df.LSV
js_df = pd.get_dummies(js_df)

covariates = ['D1854xLSV', 'D1854', 'LSV'] + [x for x in js_df.columns.values if 'area' in x]
model.fit(js_df[covariates], js_df.death)
print(f'treatment:{model.coef_[0]}')

# result
# treatment:-101.22916666666664

# log を取って回帰するとどうなるか
model.fit(js_df[covariates], js_df.death.apply(lambda x: log(x)))
print(f'treatment (log):{model.coef_[0]}')

# result
# treatment (log):-0.5661016588136917
