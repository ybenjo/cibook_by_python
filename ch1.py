# wget http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
from scipy import stats
from load_male_data import load_male_data, load_biased_male_data


male_data = load_male_data()
# 1.4.2
# calc conversion_rate / avg spend per treatment
for t in [0, 1]:
    subset = [e for e in male_data if e['treatment'] == t]
    conversion_rate = sum(e['conversion'] for e in subset) / len(subset)
    avg_spend = sum(e['spend'] for e in subset) / len(subset)
    print(f'treatment:{t},CVR:{conversion_rate},Avg.spend:{avg_spend}')

# result
# treatment:0,CVR:0.005726086548390125,Avg.spend:0.6527893551112363
# treatment:1,CVR:0.01253109306800582,Avg.spend:1.4226165110057738

# run t-test
print(stats.ttest_ind(
    [e['spend'] for e in male_data if e['treatment'] == 0],
    [e['spend'] for e in male_data if e['treatment'] == 1],
    equal_var=True
    ))

# result
# Ttest_indResult(statistic=-5.300090294465472, pvalue=1.163200872605869e-07)

biased_male_data = load_biased_male_data()
for t in [0, 1]:
    subset = [e for e in biased_male_data if e['treatment'] == t]
    conversion_rate = sum(e['conversion'] for e in subset) / len(subset)
    avg_spend = sum(e['spend'] for e in subset) / len(subset)
    print(f'treatment:{t},CVR:{conversion_rate},Avg.spend:{avg_spend}')

# result
# treatment:0,CVR:0.005047785704670884,Avg.spend:0.6443215776012919
# treatment:1,CVR:0.013459185457087922,Avg.spend:1.5728876070617035

# run t-test
print(stats.ttest_ind(
    [e['spend'] for e in biased_male_data if e['treatment'] == 0],
    [e['spend'] for e in biased_male_data if e['treatment'] == 1],
    equal_var=True
    ))

# result
# Ttest_indResult(statistic=-5.175501350096099, pvalue=2.2867019524693984e-07)
