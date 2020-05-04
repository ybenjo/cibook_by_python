from math import log
from collections import Counter
import pyreadr
from sklearn.linear_model import LinearRegression

cigar = pyreadr.read_r('./data/Cigar.rdata')['Cigar']
skip_states = [3, 9, 10, 22, 21, 23, 31, 33, 48]

cigar['skip_state'] = cigar.state.apply(lambda x: 1 if x in skip_states else 0)
cigar = cigar[(cigar.skip_state == 0) & (cigar.year >= 70)]
cigar['area'] = cigar.state.apply(lambda x: 'CA' if x == 5 else 'Rest of US')
# カリフォルニアか否かの変数
cigar['ca'] = cigar.area.apply(lambda x: 1 if x == 'CA' else 0)
# 介入の変数
cigar['post'] = cigar.year.apply(lambda x: 1 if x > 87 else 0)

# groupby 取れないのでここから pandas を捨てる
x_patterns = set()
populations = Counter()
total_sales = Counter()
for _, e in cigar.iterrows():
    year = e.year
    ca = e.ca
    post = e.post
    pop = e.pop16
    sales_per_pop = e.sales
    sales = pop * sales_per_pop
    populations[(year, ca)] += pop
    total_sales[(year, ca)] += sales
    # ca*past is treatment!
    pattern = (ca * post, ca, post, year)
    x_patterns.add(pattern)

uniq_years = list(set(x[-1] for x in x_patterns))
X = []
y = []
for pattern in x_patterns:
    # exclude year
    year = pattern[-1]
    ca = pattern[1]
    sales = total_sales[(year, ca)] / populations[(year, ca)]
    y.append(sales)
    fv = list(pattern)[:-1]
    # add dummy year
    year_dummy = [0] * len(uniq_years)
    year_dummy[uniq_years.index(year)] = 1
    fv += year_dummy
    X.append(fv)

model = LinearRegression()
model.fit(X, y)
print(f'treatment:{model.coef_[0]}')

# result
# treatment:-20.54351862181177

model.fit(X, [log(v) for v in y])
print(f'treatment (log):{model.coef_[0]}')

# result
# treatment (log):-0.2530445100355495
