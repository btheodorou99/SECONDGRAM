import torch
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

SEED = 4

df = pd.read_csv('/data/CARD_AA/data/2023_06_AD_Imaging/UKBiobank/2023_06_extractedFeatures/ukb672504_imaging.csv', low_memory=False)
df = df[['eid', 'sex_f31_0_0', 'age_at_recruitment_f21022_0_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0'] + [c for c in df.columns if 'T1Hier' in c]]
df = df[df['T1Hier_vol_hemisphere_lefthemispheres_2'].notnull()]
df = df[df['townsend_deprivation_index_at_recruitment_f22189_0_0'].notnull()]

dfStatic = df[['eid', 'sex_f31_0_0', 'age_at_recruitment_f21022_0_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0']]
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

scaleStatic = MinMaxScaler()
oneHotStatic = OneHotEncoder(sparse_output=False)
dfStatic[['age_at_recruitment_f21022_0_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0']] = scaleStatic.fit_transform(dfStatic[['age_at_recruitment_f21022_0_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0']])
staticOneHot = oneHotStatic.fit_transform(dfStatic[['sex_f31_0_0']])
dfStatic = pd.concat([dfStatic[['eid', 'age_at_recruitment_f21022_0_0', 'townsend_deprivation_index_at_recruitment_f22189_0_0']], pd.DataFrame(staticOneHot, columns=oneHotStatic.get_feature_names_out())], axis=1)
scaleFirst = MinMaxScaler()
dfFirst[[c for c in dfFirst.columns if c.endswith('_2')]] = scaleFirst.fit_transform(dfFirst[[c for c in dfFirst.columns if c.endswith('_2')]])
scaleSecond = MinMaxScaler()
dfSecond[[c for c in dfSecond.columns if c.endswith('_3')]] = scaleSecond.fit_transform(dfSecond[[c for c in dfSecond.columns if c.endswith('_3')]])
pickle.dump(scaleStatic, open('/data/theodoroubp/imageGen/scaleStatic.pkl', 'wb'))
pickle.dump(oneHotStatic, open('/data/theodoroubp/imageGen/oneHotStatic.pkl', 'wb'))
pickle.dump(scaleFirst, open('/data/theodoroubp/imageGen/scaleFirst.pkl', 'wb'))
pickle.dump(scaleSecond, open('/data/theodoroubp/imageGen/scaleSecond.pkl', 'wb'))

staticFeatures = {row.eid: row.tolist()[1:] for _, row in dfStatic.iterrows()}
firstFeatures = {row.eid: row.tolist()[1:] for _, row in dfFirst.iterrows()}
secondFeatures = {row.eid: row.tolist()[1:] for _, row in dfSecond.iterrows()}

data = [(torch.FloatTensor(staticFeatures[e]), torch.FloatTensor(firstFeatures[e]), torch.FloatTensor(secondFeatures[e]) if e in secondFeatures else None) for e in staticFeatures if e==e and e in firstFeatures]
trainData, testData = train_test_split(data, test_size=0.2)
trainData, valData = train_test_split(trainData, test_size=0.1)
pickle.dump(trainData, open('/data/theodoroubp/imageGen/trainData.pkl', 'wb'))
pickle.dump(valData, open('/data/theodoroubp/imageGen/valData.pkl', 'wb'))
pickle.dump(testData, open('/data/theodoroubp/imageGen/testData.pkl', 'wb'))

for (k, d) in [('Overall', data), ('Train', trainData), ('Validation', valData), ('Test', testData)]:
    print(k)
    print(f'\tDataset Size: {len(d)}')
    print(f'\tNumber of Second Images: {len([p for p in d if d[2] is not None])}')
    print()
print(f'Demographic Dimensionality: {len(data[0][0])}')
print(f'Demographic Dimensionality: {len(data[0][1])}')
