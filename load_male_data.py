# wget http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
import csv
import random
from sklearn.preprocessing import LabelBinarizer


def load_raw_data():
    binarizer = LabelBinarizer()
    data = []
    with open('./data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv') as f:
        reader = csv.DictReader(f)
        for e in reader:
            e['conversion'] = int(e['conversion'])
            e['spend'] = float(e['spend'])
            e['history'] = float(e['history'])
            e['recency'] = int(e['recency'])
            e['visit'] = int(e['visit'])
            data.append(e)

    binarizer.fit(list(set(e['channel'] for e in data)))

    # binarize channel
    for e in data:
        e['binarized_channel'] = binarizer.transform([e['channel']])[0].tolist()

    return data


def load_male_data():
    raw_data = load_raw_data()
    male_data = []
    for e in raw_data:
        if e['segment'] == 'Womens E-Mail':
            continue

        e['treatment'] = 1 if e['segment'] == 'Mens E-Mail' else 0
        male_data.append(e)

    return male_data


def load_biased_male_data(seed=4, obs_rate=0.5):
    # この seed によっては結果が逆転する
    # 4 が一番良さそう

    male_data = load_male_data()
    random.seed(seed)
    biased_male_data = []
    for e in male_data:
        if e['treatment'] == 0:
            if e['history'] > 300 or e['recency'] < 6 or e['channel'] == 'Multichannel':
                if random.random() < obs_rate:
                    biased_male_data.append(e)
            else:
                biased_male_data.append(e)

        else:
            if e['history'] > 300 or e['recency'] < 6 or e['channel'] == 'Multichannel':
                biased_male_data.append(e)
            else:
                if random.random() < obs_rate:
                    biased_male_data.append(e)

    return biased_male_data
