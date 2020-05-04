# wget http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
from sklearn.linear_model import LinearRegression
from load_male_data import load_male_data, load_biased_male_data

male_data = load_male_data()
biased_male_data = load_biased_male_data()

# 2.1.5
model = LinearRegression()

model.fit([[e['treatment'], e['history']] for e in biased_male_data],
          [e['spend'] for e in biased_male_data])

b_treatment = model.coef_[0]
print(f'biased treatment:{b_treatment}')
# result
# 0.8493028658202932

# non-biased data's beta_treatment
model.fit([[e['treatment'], e['history']] for e in male_data],
          [e['spend'] for e in male_data])

b_treatment = model.coef_[0]
print(f'true treatment:{b_treatment}')
# result
# 0.7674496193131057

# 2.2.1
# RCT data
model.fit([[e['treatment']] for e in male_data],
          [e['spend'] for e in male_data])

b_treatment = model.coef_[0]
print(f'RCT treatment:{b_treatment}')
# result
# 0.7698271558945355

# biased data
model.fit([[e['treatment']] for e in biased_male_data],
          [e['spend'] for e in biased_male_data])

b_treatment = model.coef_[0]
print(f'biased (naive) treatment:{b_treatment}')
# result
# 0.928566029460413

# consider recency/history/channel
model.fit([[e['treatment'], e['history'], e['recency']] + e['binarized_channel'] for e in biased_male_data],
          [e['spend'] for e in biased_male_data])

b_treatment = model.coef_[0]
print(f'regression treatment:{b_treatment}')
# result
# 0.7969309003849526
# RCT の単回帰の結果に近づきましたね

# 2.2.3
reg_a = LinearRegression()
reg_a.fit([[e['treatment'], e['recency']] + e['binarized_channel'] for e in biased_male_data],
          [e['spend'] for e in biased_male_data])

reg_b = LinearRegression()
reg_b.fit([[e['treatment'], e['recency']] + e['binarized_channel'] + [e['history']] for e in biased_male_data],
          [e['spend'] for e in biased_male_data])

reg_c = LinearRegression()
reg_c.fit([[e['treatment'], e['recency']] + e['binarized_channel'] for e in biased_male_data],
          [e['history'] for e in biased_male_data])

# ovb = reg_c.treatment * reg_b.history
ovb = reg_c.coef_[0] * reg_b.coef_[-1]
# テキストだと符号がおかしいけれど
# history を omit したモデル reg_a における treatment は真の treatment reg_b に ovb が乗ったものである
# つまり reg_a.treatment = reg_b.treatment + ovb
print(f'alpha_1 - beta_1:{reg_a.coef_[0] - reg_b.coef_[0]}')
print(f'ovb:{ovb}')
# result
# alpha_1 - beta_1:0.03259534581385537
# ovb:0.03259534581385516

# 2.2.7
model = LinearRegression()
model.fit([[e['visit'], e['recency']] + e['binarized_channel'] + [e['history']] for e in biased_male_data],
          [e['treatment'] for e in biased_male_data])

print(f'coef of visit:{model.coef_[0]}')
# result
# coef of visit:0.1510981090071298

# estimate treatment
model = LinearRegression()
model.fit([[e['treatment'], e['visit'], e['recency']] + e['binarized_channel'] + [e['history']] for e in biased_male_data],
          [e['spend'] for e in biased_male_data])

print(f'treatment (by adding visit variable):{model.coef_[0]}')
# result
# treatment (by adding visit variable):0.1820430067582668
