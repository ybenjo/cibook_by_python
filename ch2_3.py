import pyreadr
# from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from tqdm import tqdm

# download from https://github.com/itamarcaspi/experimentdatar/blob/master/data/vouchers.rda
orig_data = pyreadr.read_r('./data/vouchers.rda')['vouchers']
# 定数項を入れないと satatsmodels の ols が定数項を加味してくれない
orig_data['CONSTANT'] = 1

# 論文の条件で抽出
regression_data = orig_data[(orig_data.TAB3SMPL == 1) & (orig_data.BOG95SMP == 1)]

# 介入を示す変数
z = ['VOUCH0']

# 共変量
covariates = ['SVY', 'HSVISIT', 'AGE']
covariates += [f'STRATA{x}' for x in range(1, 7)]
covariates += ['D1993', 'D1995', 'D1997']
covariates += [f'DMONTH{x}' for x in range(1, 13)]
covariates += ['SEX2']

# 目的変数
targets = ['TOTSCYRS', 'INSCHL', 'PRSCH_C', 'USNGSCH', 'PRSCHA_1']
targets += [f'FINISH{x}' for x in range(6, 9)]
targets += ['REPT6', 'REPT', 'NREPT', 'MARRIED', 'HASCHILD', 'HOURSUM', 'WORKING3']

# まずは z だけで y を回帰
models = {}
for y in tqdm(targets, ascii=True, desc='regression only z'):
    # 実は MARRIED の一部は NA が入っており， R の lm のデフォルトは NA が含む行を無視する
    # これバイアス的にいいのかな
    # そこで回帰のたびに omit しましょう
    # NaN の判定
    _data = regression_data[regression_data[y] == regression_data[y]]
    ols = sm.OLS(_data[[y]], _data[z + ['CONSTANT']])
    model = ols.fit()
    models[(y, 'base')] = model

# 共変量で回帰する
for y in tqdm(targets, ascii=True, desc='regression w/ covariates'):
    _data = regression_data[regression_data[y] == regression_data[y]]
    ols = sm.OLS(_data[[y]], _data[z + covariates + ['CONSTANT']])
    model = ols.fit()
    models[(y, 'covariate')] = model

# 2.3.3 通学率と奨学金の関係
# PRSCHA_1 と USNGSCH を比較しましょう
for y in ['PRSCHA_1', 'USNGSCH']:
    for _type in ['base', 'covariate']:
        print(f'{y}_{_type}:{models[(y, _type)].params[0]}')

# result
# PRSCHA_1_base:0.06294674088268509
# PRSCHA_1_covariate:0.0574307222373074
# USNGSCH_base:0.5088724640326051
# USNGSCH_covariate:0.5041599567931954

# 2.3.4 割引券は留年率を減らしているか
# model.HC0_se で stderr を取得して 1.96 足し引きする
for y in ['FINISH6', 'FINISH7', 'FINISH8', 'INSCHL', 'NREPT', 'PRSCH_C', 'REPT', 'REPT6']:
    model = models[(y, 'covariate')]
    w = model.params[0]
    sd = model.HC0_se[0]
    w_min = w - sd * 1.96
    w_max = w + sd * 1.96

    print(f'{y}_covariate: {w_min} <-> {w} <-> {w_max}')

# FINISH6_covariate: 0.0003888292994237638 <-> 0.022945967851745554 <-> 0.045503106404067344
# FINISH7_covariate: -0.007172087549380188 <-> 0.03074834454123925 <-> 0.06866877663185869
# FINISH8_covariate: 0.04769485913527166 <-> 0.10020409189989303 <-> 0.1527133246645144
# INSCHL_covariate: -0.0321760936496368 <-> 0.006903514144260429 <-> 0.04598312193815766
# EPT_covariate: -0.12005977904094013 <-> -0.06667541929923938 <-> -0.013291059557538634
# PRSCH_C_covariate: 0.09957958678636933 <-> 0.15333001006405259 <-> 0.20708043334173584
# REPT_covariate: -0.09997122309448322 <-> -0.0547603184203557 <-> -0.00954941374622817
# REPT6_covariate: -0.10649597125384064 <-> -0.05936863016149352 <-> -0.012241289069146392

# 2.3.5
data_tbl4_bog95 = orig_data[(orig_data.TAB3SMPL == 1) & (orig_data.BOG95SMP == 1)]
not_nan_columns = ['SCYFNSH', 'PRSCHA_1', 'REPT6', 'NREPT', 'INSCHL']
not_nan_columns += [f'FINISH{x}' for x in range(6, 9)]
not_nan_columns += ['PRSCH_C', 'PRSCHA_2', 'TOTSCYRS', 'REPT']

for c in not_nan_columns:
    # reject NaN
    data_tbl4_bog95 = data_tbl4_bog95[data_tbl4_bog95[c] == data_tbl4_bog95[c]]

# columns = ['SVY', 'CONSTANT', 'HSVISIT', 'DJAMUNDI', 'PHONE', 'AGE']
# columns += [f'STRATA{x}' for x in range(1, 7)]
# columns += ['STRATAMS', 'DBOGOTA', 'D1993', 'D1995', 'D1997']
# columns += [f'DMONTH{x}' for x in range(1, 13)]
# columns += ['SEX_MISS']
# columns += [f'FINISH{x}' for x in range(6, 9)]
# columns += ['REPT6', 'REPT', 'NREPT', 'SEX2', 'TOTSCYRS', 'MARRIED', 'HASCHILD']
# columns += ['HOURSUM', 'WORKING3', 'INSCHL', 'PRSCH_C', 'USNGSCH', 'PRSCHA_1']

# select only female
regression_data = data_tbl4_bog95[data_tbl4_bog95.SEX2 == 0]
for y in tqdm(targets, ascii=True, desc='regression w/ covariates (only female)'):
    _data = regression_data[regression_data[y] == regression_data[y]]
    ols = sm.OLS(_data[[y]], _data[z + covariates + ['CONSTANT']])
    model = ols.fit()
    models[(y, 'covariate', 'female')] = model

# select only male
regression_data = data_tbl4_bog95[data_tbl4_bog95.SEX2 == 1]
for y in tqdm(targets, ascii=True, desc='regression w/ covariates (only female)'):
    _data = regression_data[regression_data[y] == regression_data[y]]
    ols = sm.OLS(_data[[y]], _data[z + covariates + ['CONSTANT']])
    model = ols.fit()
    models[(y, 'covariate', 'male')] = model

# figure 2.5
for sex in ['female', 'male']:
    for y in ['PRSCHA_1', 'USNGSCH']:
        model = models[(y, 'covariate', sex)]
        w = model.params[0]
        sd = model.HC0_se[0]
        w_min = w - sd * 1.96
        w_max = w + sd * 1.96

        print(f'{sex}:{y}_covariate: {w_min} <-> {w} <-> {w_max}')

    print('-------------------------------------------------')

# result
# female:PRSCHA_1_covariate: -0.020143104883902804 <-> 0.022858521054893822 <-> 0.06586014699369044
# female:USNGSCH_covariate: 0.48254104474319187 <-> 0.5435997462487114 <-> 0.6046584477542309
# male:PRSCHA_1_covariate: 0.03956317285890743 <-> 0.09017184794031903 <-> 0.14078052302173064
# male:USNGSCH_covariate: 0.405233681694163 <-> 0.4676729112164002 <-> 0.5301121407386373

# figure 2.6
for sex in ['female', 'male']:
    for y in ['FINISH6', 'FINISH7', 'FINISH8', 'INSCHL', 'NREPT', 'PRSCH_C', 'REPT', 'REPT6', 'TOTSCYRS']:
        model = models[(y, 'covariate', sex)]
        w = model.params[0]
        sd = model.HC0_se[0]
        w_min = w - sd * 1.96
        w_max = w + sd * 1.96

        print(f'{sex}:{y}_covariate: {w_min} <-> {w} <-> {w_max}')

    print('-------------------------------------------------')

# result
# female:FINISH6_covariate: 0.006564986305322453 <-> 0.031746375791598155 <-> 0.056927765277873854
# female:FINISH7_covariate: -0.0071239807550086895 <-> 0.04114470179824897 <-> 0.08941338435150664
# female:FINISH8_covariate: 0.03384200029839929 <-> 0.10473067557232467 <-> 0.17561935084625005
# female:INSCHL_covariate: -0.0186699377732938 <-> 0.03471632814285565 <-> 0.0881025940590051
# female:NREPT_covariate: -0.09642381300568886 <-> -0.03133560245411438 <-> 0.03375260809746009
# female:PRSCH_C_covariate: 0.09484588549570633 <-> 0.17105216095643372 <-> 0.2472584364171611
# female:REPT_covariate: -0.08988554061224435 <-> -0.028959785779132656 <-> 0.03196596905397905
# female:REPT6_covariate: -0.09473460495242172 <-> -0.03616691164526466 <-> 0.02240078166189241
# female:TOTSCYRS_covariate: -0.03232060348469547 <-> 0.09092168042791371 <-> 0.21416396434052287
# -------------------------------------------------
# male:FINISH6_covariate: -0.021177865854108857 <-> 0.01444924620585804 <-> 0.05007635826582493
# male:FINISH7_covariate: -0.030819875824427252 <-> 0.026427596464665637 <-> 0.08367506875375852
# male:FINISH8_covariate: 0.017571747675130464 <-> 0.09498078757032702 <-> 0.17238982746552356
# male:INSCHL_covariate: -0.07610435904470145 <-> -0.019508539890272812 <-> 0.037087279264155816
# male:NREPT_covariate: -0.18403882465843058 <-> -0.1014750148427504 <-> -0.018911205027070208
# male:PRSCH_C_covariate: 0.06036453698580779 <-> 0.13634677289825592 <-> 0.21232900881070405
# male:REPT_covariate: -0.14902559723437303 <-> -0.08302440311777091 <-> -0.017023209001168813
# male:REPT6_covariate: -0.15931573892348053 <-> -0.08661288018834999 <-> -0.013910021453219457
# male:TOTSCYRS_covariate: -0.17859270588963402 <-> -0.028628666749175948 <-> 0.12133537239128214
# -------------------------------------------------

# figure 2.7
for sex in ['female', 'male']:
    model = models[('HOURSUM', 'covariate', sex)]
    w = model.params[0]
    sd = model.HC0_se[0]
    w_min = w - sd * 1.96
    w_max = w + sd * 1.96

    print(f'{sex}:HOURSUM_covariate: {w_min} <-> {w} <-> {w_max}')
    print('-------------------------------------------------')

# result
# female:HOURSUM_covariate: -3.3950682562333343 <-> -2.115783364228365 <-> -0.8364984722233961
# -------------------------------------------------
# male:HOURSUM_covariate: -2.7393994586444643 <-> -0.6376087493230705 <-> 1.4641819599983235
# -------------------------------------------------
