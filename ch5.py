from math import log
from load_male_data import load_raw_data
from sklearn.linear_model import LinearRegression

raw_data = load_raw_data()
male_data = []
for e in raw_data:
    if e['segment'] not in ['Mens E-Mail', 'No E-Mail']:
        continue

    e['treatment'] = 1 if e['segment'] == 'Mens E-Mail' else 0
    e['history_log'] = log(e['history'])
    male_data.append(e)

threshold_value = 5.5
rdd_data = [e for e in male_data if e['history_log'] >= threshold_value and e['treatment'] == 1]
rdd_data += [e for e in male_data if e['history_log'] < threshold_value and e['treatment'] == 0]

# 平均来訪率の計算
for z in [0, 1]:
    _data = [e for e in rdd_data if e['treatment'] == z]
    print(f"treatment={z}:{sum(e['visit'] for e in _data) / len(_data) * 100}%")

# result
# treatment=0:9.06936665230504%
# treatment=1:22.40021721422753%

model = LinearRegression()
model.fit([[e['treatment'], e['history_log']] for e in rdd_data], [e['visit'] for e in rdd_data])
print(f'LATE treatment:{model.coef_[0]}')

# result
# LATE treatment:0.11367198053261053
