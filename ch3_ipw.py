from sklearn.linear_model import LogisticRegression, LinearRegression
from load_male_data import load_biased_male_data


biased_male_data = load_biased_male_data()

# 3.2.2
# まずは傾向スコアを求める
model = LogisticRegression()
model.fit([[e['history'], e['recency']] + e['binarized_channel'] for e in biased_male_data],
          [e['treatment'] for e in biased_male_data])

propensity_scores = model.predict_proba([[e['history'], e['recency']] + e['binarized_channel'] for e in biased_male_data])[:, 1]
for pos, e in enumerate(biased_male_data):
    ps = propensity_scores[pos]
    weight = 1.0 / ps
    # 逆数を IPW 推定量として保存
    e['ipw'] = weight

model = LinearRegression()
model.fit([[e['treatment']]  for e in biased_male_data],
          [e['spend'] for e in biased_male_data],
          # 回帰時に sample_weight で指定しましょう
          sample_weight=[e['ipw'] for e in biased_male_data])

print(f'treatment:{model.coef_[0]}')
# result
# treatment:0.8998412323190048
