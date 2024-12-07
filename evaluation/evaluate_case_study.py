import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config

# Volume is in mm^3, area is in mm^2

sncVol = ['vol_mtg_sn_snc_leftcit168',
'vol_mtg_sn_snc_rightcit168',
'vol_mtg_sn_snc_leftsnseg',
'vol_mtg_sn_snc_rightsnseg',
'vol_mtg_sn_snc_leftdeep_cit168',
'vol_mtg_sn_snc_rightdeep_cit168']

sncArea = ['area_mtg_sn_snc_leftcit168',
'area_mtg_sn_snc_rightcit168',
'area_mtg_sn_snc_leftsnseg',
'area_mtg_sn_snc_rightsnseg',
'area_mtg_sn_snc_leftdeep_cit168',
'area_mtg_sn_snc_rightdeep_cit168']

config = Config()
NUM_RUNS = config.num_runs

staticMap = pickle.load(open('/home/SECONDGRAM/data/staticMap.pkl', 'rb'))
featureMap = pickle.load(open('/home/SECONDGRAM/data/featureMap.pkl', 'rb'))
scaleFirst = pickle.load(open('/home/SECONDGRAM/data/scaleFirst.pkl', 'rb'))
scaleSecond = pickle.load(open('/home/SECONDGRAM/data/scaleSecond.pkl', 'rb'))

keys = ['real', 'secondgram']
realData = pickle.load(open('/home/SECONDGRAM/data/trainData.pkl', 'rb')) + pickle.load(open('/home/SECONDGRAM/data/valData.pkl', 'rb')) + pickle.load(open('/home/SECONDGRAM/data/testData.pkl', 'rb'))

results = {}
for k in tqdm(keys):
    keyResults = {}
    if k == 'real':
        data = [p for p in realData if p[2] is not None]
    else:
        data = []
        for i in range(1, NUM_RUNS+1):
            generatedData = pickle.load(open(f'/home/SECONDGRAM/generations/generatedTrainData_{k}_{i}.pkl', 'rb')) + pickle.load(open(f'/home/SECONDGRAM/generations/generatedValData_{k}_{i}.pkl', 'rb')) + pickle.load(open(f'/home/SECONDGRAM/generations/generatedTestData_{k}_{i}.pkl', 'rb'))
            data += [(realData[i][0], realData[i][1], generatedData[i]) for i in range(len(realData)) if realData[i][2] is not None]

    data = [(p[0], scaleFirst.inverse_transform(p[1].reshape(1, -1)).squeeze(), scaleSecond.inverse_transform(p[2].reshape(1, -1)).squeeze()) for p in data]
    park = [p for p in data if p[0][staticMap['PARKINSONISM']] == 1]
    noPark = [p for p in data if p[0][staticMap['PARKINSONISM']] == 0]

    columnResults = {}
    for c in sncVol:
        columnResults[c] = {}
        columnResults[c]['Parkinsonism Initial Value'] = np.mean([p[1][featureMap[c]] for p in park])
        columnResults[c]['Parkinsonism Final Value'] = np.mean([p[2][featureMap[c]] for p in park])
        columnResults[c]['No Parkinsonism Initial Value'] = np.mean([p[1][featureMap[c]] for p in noPark])
        columnResults[c]['No Parkinsonism Final Value'] = np.mean([p[2][featureMap[c]] for p in noPark])
        columnResults[c]['Parkinsonism Change'] = np.mean([p[2][featureMap[c]] - p[1][featureMap[c]] for p in park])
        columnResults[c]['No Parkinsonism Change'] = np.mean([p[2][featureMap[c]] - p[1][featureMap[c]] for p in noPark])
        columnResults[c]['Parkinsonism Change %'] = np.mean([(p[2][featureMap[c]] - p[1][featureMap[c]]) / p[1][featureMap[c]] for p in park])
        columnResults[c]['No Parkinsonism Change %'] = np.mean([(p[2][featureMap[c]] - p[1][featureMap[c]]) / p[1][featureMap[c]] for p in noPark])
        
    for c in sncArea:
        columnResults[c] = {}
        columnResults[c]['Parkinsonism Initial Value'] = np.mean([p[1][featureMap[c]] for p in park])
        columnResults[c]['Parkinsonism Final Value'] = np.mean([p[2][featureMap[c]] for p in park])
        columnResults[c]['No Parkinsonism Initial Value'] = np.mean([p[1][featureMap[c]] for p in noPark])
        columnResults[c]['No Parkinsonism Final Value'] = np.mean([p[2][featureMap[c]] for p in noPark])
        columnResults[c]['Parkinsonism Change'] = np.mean([p[2][featureMap[c]] - p[1][featureMap[c]] for p in park])
        columnResults[c]['No Parkinsonism Change'] = np.mean([p[2][featureMap[c]] - p[1][featureMap[c]] for p in noPark])
        columnResults[c]['Parkinsonism Change %'] = np.mean([(p[2][featureMap[c]] - p[1][featureMap[c]]) / p[1][featureMap[c]] for p in park])
        columnResults[c]['No Parkinsonism Change %'] = np.mean([(p[2][featureMap[c]] - p[1][featureMap[c]]) / p[1][featureMap[c]] for p in noPark])


    keyResults['Parkinsonism Initial Volume'] = np.mean([columnResults[c]['Parkinsonism Initial Value'] for c in sncVol])
    keyResults['Parkinsonism Final Volume'] = np.mean([columnResults[c]['Parkinsonism Final Value'] for c in sncVol])
    keyResults['No Parkinsonism Initial Volume'] = np.mean([columnResults[c]['No Parkinsonism Initial Value'] for c in sncVol])
    keyResults['No Parkinsonism Final Volume'] = np.mean([columnResults[c]['No Parkinsonism Final Value'] for c in sncVol])
    keyResults['Parkinsonism Change Volume'] = np.mean([columnResults[c]['Parkinsonism Change'] for c in sncVol])
    keyResults['No Parkinsonism Change Volume'] = np.mean([columnResults[c]['No Parkinsonism Change'] for c in sncVol])
    keyResults['Parkinsonism Change % Volume'] = np.mean([columnResults[c]['Parkinsonism Change %'] for c in sncVol])
    keyResults['No Parkinsonism Change % Volume'] = np.mean([columnResults[c]['No Parkinsonism Change %'] for c in sncVol])

    keyResults['Parkinsonism Initial Area'] = np.mean([columnResults[c]['Parkinsonism Initial Value'] for c in sncArea])
    keyResults['Parkinsonism Final Area'] = np.mean([columnResults[c]['Parkinsonism Final Value'] for c in sncArea])
    keyResults['No Parkinsonism Initial Area'] = np.mean([columnResults[c]['No Parkinsonism Initial Value'] for c in sncArea])
    keyResults['No Parkinsonism Final Area'] = np.mean([columnResults[c]['No Parkinsonism Final Value'] for c in sncArea])
    keyResults['Parkinsonism Change Area'] = np.mean([columnResults[c]['Parkinsonism Change'] for c in sncArea])
    keyResults['No Parkinsonism Change Area'] = np.mean([columnResults[c]['No Parkinsonism Change'] for c in sncArea])
    keyResults['Parkinsonism Change % Area'] = np.mean([columnResults[c]['Parkinsonism Change %'] for c in sncArea])
    keyResults['No Parkinsonism Change % Area'] = np.mean([columnResults[c]['No Parkinsonism Change %'] for c in sncArea])

    keyResults['Parkinsonism Inital Value'] = np.mean([keyResults['Parkinsonism Initial Volume'], keyResults['Parkinsonism Initial Area']])
    keyResults['Parkinsonism Final Value'] = np.mean([keyResults['Parkinsonism Final Volume'], keyResults['Parkinsonism Final Area']])
    keyResults['No Parkinsonism Initial Value'] = np.mean([keyResults['No Parkinsonism Initial Volume'], keyResults['No Parkinsonism Initial Area']])
    keyResults['No Parkinsonism Final Value'] = np.mean([keyResults['No Parkinsonism Final Volume'], keyResults['No Parkinsonism Final Area']])
    keyResults['Parkinsonism Change'] = np.mean([keyResults['Parkinsonism Change Volume'], keyResults['Parkinsonism Change Area']])
    keyResults['No Parkinsonism Change'] = np.mean([keyResults['No Parkinsonism Change Volume'], keyResults['No Parkinsonism Change Area']])
    keyResults['Parkinsonism Change %'] = np.mean([keyResults['Parkinsonism Change % Volume'], keyResults['Parkinsonism Change % Area']])
    keyResults['No Parkinsonism Change %'] = np.mean([keyResults['No Parkinsonism Change % Volume'], keyResults['No Parkinsonism Change % Area']])

    keyResults['Column Results'] = columnResults
    results[k] = keyResults

pickle.dump(results, open('/home/SECONDGRAM/stats/caseStudyStats.pkl', 'wb'))

for k in results:
    results[k].pop('Column Results')

df = pd.DataFrame(results)
df = df.T
df.to_csv('/home/SECONDGRAM/stats/caseStudyStats.csv')



# # SUBSTANTIA NIGRA
# ['vol_mtg_sn_snc_leftcit168',
# 'vol_mtg_sn_snc_rightcit168',
# 'vol_mtg_sn_snr_leftcit168',
# 'vol_mtg_sn_snr_rightcit168',
# 'vol_mtg_sn_snc_leftsnseg',
# 'vol_mtg_sn_snc_rightsnseg',
# 'vol_mtg_sn_snr_leftsnseg',
# 'vol_mtg_sn_snr_rightsnseg',
# 'vol_mtg_sn_snc_leftdeep_cit168',
# 'vol_mtg_sn_snc_rightdeep_cit168',
# 'vol_mtg_sn_snr_leftdeep_cit168',
# 'vol_mtg_sn_snr_rightdeep_cit168']

# ['area_mtg_sn_snc_leftcit168',
# 'area_mtg_sn_snc_rightcit168',
# 'area_mtg_sn_snr_leftcit168',
# 'area_mtg_sn_snr_rightcit168',
# 'area_mtg_sn_snc_leftsnseg',
# 'area_mtg_sn_snc_rightsnseg',
# 'area_mtg_sn_snr_leftsnseg',
# 'area_mtg_sn_snr_rightsnseg',
# 'area_mtg_sn_snc_leftdeep_cit168',
# 'area_mtg_sn_snc_rightdeep_cit168',
# 'area_mtg_sn_snr_leftdeep_cit168',
# 'area_mtg_sn_snr_rightdeep_cit168']

# ['thk_mtg_sn_snc_leftcit168',
# 'thk_mtg_sn_snc_rightcit168',
# 'thk_mtg_sn_snr_leftcit168',
# 'thk_mtg_sn_snr_rightcit168',
# 'thk_mtg_sn_snc_leftsnseg',
# 'thk_mtg_sn_snc_rightsnseg',
# 'thk_mtg_sn_snr_leftsnseg',
# 'thk_mtg_sn_snr_rightsnseg',
# 'thk_mtg_sn_snc_leftdeep_cit168',
# 'thk_mtg_sn_snc_rightdeep_cit168',
# 'thk_mtg_sn_snr_leftdeep_cit168',
# 'thk_mtg_sn_snr_rightdeep_cit168']

# # WHITE MATTER
# ['vol_left_cerebellum_white_matterdktregions',
# 'vol_right_cerebellum_white_matterdktregions',
# 'area_left_cerebellum_white_matterdktregions',
# 'area_right_cerebellum_white_matterdktregions',
# 'thk_left_cerebellum_white_matterdktregions',
# 'thk_right_cerebellum_white_matterdktregions']

# # BASAL GANGLIA
# ['vol_bn_gp_gpe_leftcit168',
# 'vol_bn_gp_gpe_rightcit168',
# 'vol_bn_gp_gpi_leftcit168',
# 'vol_bn_gp_gpi_rightcit168',
# 'vol_bn_gp_vep_leftcit168',
# 'vol_bn_gp_vep_rightcit168',
# 'vol_bn_str_ca_leftcit168',
# 'vol_bn_str_ca_rightcit168',
# 'vol_bn_str_nac_leftcit168',
# 'vol_bn_str_nac_rightcit168',
# 'vol_bn_str_pu_leftcit168',
# 'vol_bn_str_pu_rightcit168',
# 'area_bn_gp_gpe_leftcit168',
# 'area_bn_gp_gpe_rightcit168',
# 'area_bn_gp_gpi_leftcit168',
# 'area_bn_gp_gpi_rightcit168',
# 'area_bn_gp_vep_leftcit168',
# 'area_bn_gp_vep_rightcit168',
# 'area_bn_str_ca_leftcit168',
# 'area_bn_str_ca_rightcit168',
# 'area_bn_str_nac_leftcit168',
# 'area_bn_str_nac_rightcit168',
# 'area_bn_str_pu_leftcit168',
# 'area_bn_str_pu_rightcit168',
# 'thk_bn_gp_gpe_leftcit168',
# 'thk_bn_gp_gpe_rightcit168',
# 'thk_bn_gp_gpi_leftcit168',
# 'thk_bn_gp_gpi_rightcit168',
# 'thk_bn_gp_vep_leftcit168',
# 'thk_bn_gp_vep_rightcit168',
# 'thk_bn_str_ca_leftcit168',
# 'thk_bn_str_ca_rightcit168',
# 'thk_bn_str_nac_leftcit168',
# 'thk_bn_str_nac_rightcit168',
# 'thk_bn_str_pu_leftcit168',
# 'thk_bn_str_pu_rightcit168',
# 'vol_bn_gp_gpe_leftdeep_cit168',
# 'vol_bn_gp_gpe_rightdeep_cit168',
# 'vol_bn_gp_gpi_leftdeep_cit168',
# 'vol_bn_gp_gpi_rightdeep_cit168',
# 'vol_bn_str_ca_leftdeep_cit168',
# 'vol_bn_str_ca_rightdeep_cit168',
# 'vol_bn_str_pu_leftdeep_cit168',
# 'vol_bn_str_pu_rightdeep_cit168',
# 'area_bn_gp_gpe_leftdeep_cit168',
# 'area_bn_gp_gpe_rightdeep_cit168',
# 'area_bn_gp_gpi_leftdeep_cit168',
# 'area_bn_gp_gpi_rightdeep_cit168',
# 'area_bn_str_ca_leftdeep_cit168',
# 'area_bn_str_ca_rightdeep_cit168',
# 'area_bn_str_pu_leftdeep_cit168',
# 'area_bn_str_pu_rightdeep_cit168',
# 'thk_bn_gp_gpe_leftdeep_cit168',
# 'thk_bn_gp_gpe_rightdeep_cit168',
# 'thk_bn_gp_gpi_leftdeep_cit168',
# 'thk_bn_gp_gpi_rightdeep_cit168',
# 'thk_bn_str_ca_leftdeep_cit168',
# 'thk_bn_str_ca_rightdeep_cit168',
# 'thk_bn_str_pu_leftdeep_cit168',
# 'thk_bn_str_pu_rightdeep_cit168']