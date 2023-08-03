import torch
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

SEED = 4

df = pd.read_csv('/data/CARD_AA/data/2023_06_AD_Imaging/UKBiobank/2023_06_extractedFeatures/ukb672504_imaging.csv', low_memory=False)
df = df[['eid', 'age_when_attended_assessment_centre_f21003_2_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0'] + [c for c in df.columns if 'T1Hier' in c]]
df = df[df['T1Hier_vol_hemisphere_lefthemispheres_2'].notnull()]
df = df[df['townsend_deprivation_index_at_recruitment_f22189_0_0'].notnull()]
df = df.merge(pd.read_parquet('/data/CARD_AA/data/2023_06_AD_Imaging/UKBiobank/2023_06_extractedFeatures/combinedData.parquet.gzip').drop_duplicates('eid_invicro'), left_on='eid', right_on='eid_invicro')
df['eid'] = df['eid_x']
df = df[df['AD_prs'].notnull()]
df = df[df['PD_prs'].notnull()]
df['STROKE'] = df['Date_of_STROKE'].notnull().astype(int)
df['ALL_DEMENTIA'] = df['Date_of_ALL_DEMENTIA'].notnull().astype(int)
df['PARKINSONISM'] = df['Date_of_PARKINSONISM'].notnull().astype(int)
df['SLEEP'] = df['Date_of_SLEEP'].notnull().astype(int)
df['MS'] = df['Date_of_MS'].notnull().astype(int)
df['EPILEPSY'] = df['Date_of_EPILEPSY'].notnull().astype(int)
df['MIGRAINE'] = df['Date_of_MIGRAINE'].notnull().astype(int)
df['OtMOVEMENT'] = df['Date_of_OtMOVEMENT'].notnull().astype(int)
df = df[['eid', 'Gender_invicro', 'age_when_attended_assessment_centre_f21003_2_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0', 'AD_prs', 'PD_prs', 'STROKE', 'ALL_DEMENTIA', 'PARKINSONISM', 'SLEEP', 'MS', 'EPILEPSY', 'MIGRAINE', 'OtMOVEMENT'] + [c for c in df.columns if 'T1Hier' in c]]

dfStatic = df[['eid', 'Gender_invicro', 'age_when_attended_assessment_centre_f21003_2_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0', 'AD_prs', 'PD_prs', 'STROKE', 'ALL_DEMENTIA', 'PARKINSONISM', 'SLEEP', 'MS', 'EPILEPSY', 'MIGRAINE', 'OtMOVEMENT']]
dfFirst = df[['eid'] + [c for c in df.columns if c.endswith('_2')]]
dfSecond = df[['eid'] + [c for c in df.columns if c.startswith('T1Hier')]]

impFirst = IterativeImputer(random_state=SEED, max_iter=3)
dfFirst[[c for c in df.columns if c.endswith('_2')]] = impFirst.fit_transform(dfFirst[[c for c in df.columns if c.endswith('_2')]])
dfSecond[[c for c in df.columns if c.endswith('_2')]] = dfFirst[[c for c in df.columns if c.endswith('_2')]]
dfSecond = dfSecond[dfSecond['T1Hier_vol_hemisphere_lefthemispheres_3'].notnull()]
impSecond = IterativeImputer(random_state=SEED, max_iter=3)
dfSecond[[c for c in df.columns if c.startswith('T1Hier')]] = impSecond.fit_transform(dfSecond[[c for c in df.columns if c.startswith('T1Hier')]])
dfSecond = dfSecond[['eid'] + [c for c in df.columns if c.endswith('_3')]]

pickle.dump(dfFirst, open('/data/theodoroubp/tempFirst.pkl', 'wb'))
pickle.dump(dfSecond, open('/data/theodoroubp/tempSecond.pkl', 'wb'))
pickle.dump(dfStatic, open('/data/theodoroubp/tempStatic.pkl', 'wb'))

dfFirst = pickle.load(open('/data/theodoroubp/tempFirst.pkl', 'rb'))
dfSecond = pickle.load(open('/data/theodoroubp/tempSecond.pkl', 'rb'))
dfStatic = pickle.load(open('/data/theodoroubp/tempStatic.pkl', 'rb'))

scaleStatic = MinMaxScaler(feature_range=(0,1))
dfStatic[['age_when_attended_assessment_centre_f21003_2_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0', 'AD_prs', 'PD_prs']] = scaleStatic.fit_transform(dfStatic[['age_when_attended_assessment_centre_f21003_2_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0', 'AD_prs', 'PD_prs']])
scaleFirst = MinMaxScaler(feature_range=(-1,1))
dfFirst[[c for c in dfFirst.columns if c.endswith('_2')]] = scaleFirst.fit_transform(dfFirst[[c for c in dfFirst.columns if c.endswith('_2')]])
scaleSecond = MinMaxScaler(feature_range=(-1,1))
dfSecond[[c for c in dfSecond.columns if c.endswith('_3')]] = scaleSecond.fit_transform(dfSecond[[c for c in dfSecond.columns if c.endswith('_3')]])
pickle.dump(scaleStatic, open('/data/theodoroubp/imageGen/scaleStatic.pkl', 'wb'))
pickle.dump(scaleFirst, open('/data/theodoroubp/imageGen/scaleFirst.pkl', 'wb'))
pickle.dump(scaleSecond, open('/data/theodoroubp/imageGen/scaleSecond.pkl', 'wb'))

staticFeatures = {row.eid: row.tolist()[1:] for _, row in dfStatic.iterrows()}
firstFeatures = {row.eid: row.tolist()[1:] for _, row in dfFirst.iterrows()}
secondFeatures = {row.eid: row.tolist()[1:] for _, row in dfSecond.iterrows()}

data = [(torch.FloatTensor(staticFeatures[e]), torch.FloatTensor(firstFeatures[e]), torch.FloatTensor(secondFeatures[e]) if e in secondFeatures else None) for e in staticFeatures if e==e and e in firstFeatures]
trainData, testData = train_test_split(data, test_size=0.2)
trainData, valData = train_test_split(trainData, test_size=0.1)
pickle.dump(trainData, open('/data/theodoroubp/imageGen/data/trainData.pkl', 'wb'))
pickle.dump(valData, open('/data/theodoroubp/imageGen/data/valData.pkl', 'wb'))
pickle.dump(testData, open('/data/theodoroubp/imageGen/data/testData.pkl', 'wb'))

for (k, d) in [('Overall', data), ('Train', trainData), ('Validation', valData), ('Test', testData)]:
    print(k)
    print(f'\tDataset Size: {len(d)}')
    print(f'\tNumber of Second Images: {len([p for p in d if p[2] is not None])}')
    print()
print(f'Demographic Dimensionality: {len(data[0][0])}')
print(f'Image Dimensionality: {len(data[0][1])}')

# Overall
#         Dataset Size: 41706
#         Number of Second Images: 4997

# Train
#         Dataset Size: 30027
#         Number of Second Images: 3615

# Validation
#         Dataset Size: 3337
#         Number of Second Images: 391

# Test
#         Dataset Size: 8342
#         Number of Second Images: 991

# Demographic Dimensionality: 13
# Image Dimensionality: 769

# Other Demographic Column Options (all in combinedData.parquet.gzip):
# ['Date_of_AD',
# 'Date_of_VAD',
# 'Date_of_PD',
# 'Date_of_STROKE',
# 'Date_of_MND',
# 'Date_of_FTD',
# 'Date_of_ALL_DEMENTIA',
# 'Date_of_PARKINSONISM',
# 'Date_of_SLEEP',
# 'Date_of_MS',
# 'Date_of_DYSTONIA',
# 'Date_of_EPILEPSY',
# 'Date_of_MIGRAINE',
# 'Date_of_OtMOVEMENT',
# 'Date_of_SMA',
# 'Date_of_OTHER_DEGENERATION',
# ['AD_imaging_score',
# 'PD_imaging_score',
# 'norm_AD_imaging_score',
# 'norm_PD_imaging_score']

# Date_of_AD: 33
# Date_of_VAD: 16
# Date_of_PD: 99
# Date_of_STROKE: 715
# Date_of_MND: 30
# Date_of_FTD: 5
# Date_of_ALL_DEMENTIA: 94
# Date_of_PARKINSONISM: 107
# Date_of_SLEEP: 1497
# Date_of_MS: 181
# Date_of_DYSTONIA: 80
# Date_of_EPILEPSY: 434
# Date_of_MIGRAINE: 3646
# Date_of_OtMOVEMENT: 384
# Date_of_SMA: 35
# Date_of_OTHER_DEGENERATION: 72
# AD_imaging_score: 46119
# PD_imaging_score: 46119
# norm_AD_imaging_score: 46119
# norm_PD_imaging_score: 46119


# 0: Gender_invicro
# 1: age_when_attended_assessment_centre_f21003_2_0
# 2: townsend_deprivation_index_at_recruitment_f22189_0_0
# 3: AD_prs
# 4: PD_prs
# 5: STROKE
# 6: ALL_DEMENTIA
# 7: PARKINSONISM
# 8: SLEEP
# 9: MS
# 10: EPILEPSY
# 11: MIGRAINE
# 12: OtMOVEMENT