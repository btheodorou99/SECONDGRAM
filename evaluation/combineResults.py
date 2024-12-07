import os
import pickle
import numpy as np
from tqdm import tqdm
from config import Config

config = Config()
NUM_RUNS = config.num_runs
IMAGE_DIM = len(pickle.load(open('/home/SECONDGRAM/data/testData.pkl', 'rb'))[0][1])

LABEL_IDX = [5, 7]
results = {}
for fname in [f for f in os.listdir('/home/SECONDGRAM/stats/') if f.startswith('trainingStats_')]:
    stats = pickle.load(open(f'/home/SECONDGRAM/stats/{fname}', 'rb'))
    lIdx = int(fname.split('_')[1])
    key = ('_').join(fname.split('_')[2:-1])
    run = int(fname.split('_')[-1].split('.')[0])
    if lIdx not in results:
        results[lIdx] = {}
    for k in stats:
        fullKey = k if ('Real' in k or 'First' in k) else f'{k}_{key}'
        if fullKey not in results[lIdx]:
            results[lIdx][fullKey] = {}
        results[lIdx][fullKey][run] = stats[k]
          
pickle.dump(results, open(f'/home/SECONDGRAM/stats/trainingStats.pkl', 'wb'))

final_results = {}
for lIdx in results:
    if lIdx not in final_results:
        final_results[lIdx] = {}
    for key in results[lIdx]:
        if key not in final_results[lIdx]:
            final_results[lIdx][key] = {}
        for stat in list(results[lIdx][key].values())[0].keys():
            final_results[lIdx][key][stat] = (np.mean([results[lIdx][key][run][stat] for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([results[lIdx][key][run][stat] for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))

final_results = {lIdx: {stat: {key: final_results[lIdx][key][stat] for key in final_results[lIdx]} for stat in list(final_results[lIdx].values())[0].keys()} for lIdx in final_results}
print(final_results)
pickle.dump(final_results, open(f'/home/SECONDGRAM/stats/trainingStatsFinal.pkl', 'wb'))





LABEL_IDX = [1, 2, 3, 4]
results = {}
for fname in [f for f in os.listdir('/home/SECONDGRAM/stats/') if f.startswith('unscaledTrainingStats_')]:
    stats = pickle.load(open(f'/home/SECONDGRAM/stats/{fname}', 'rb'))
    lIdx = int(fname.split('_')[1])
    key = ('_').join(fname.split('_')[2:-1])
    run = int(fname.split('_')[-1].split('.')[0])
    if lIdx not in results:
        results[lIdx] = {}
    for k in stats:
        fullKey = k if ('Real' in k or 'First' in k) else f'{k}_{key}'
        if fullKey not in results[lIdx]:
            results[lIdx][fullKey] = {}
        results[lIdx][fullKey][run] = stats[k]
          
pickle.dump(results, open(f'/home/SECONDGRAM/stats/unscaledTrainingStats.pkl', 'wb'))

final_results = {}
for lIdx in results:
    if lIdx not in final_results:
        final_results[lIdx] = {}
    for key in results[lIdx]:
        if key not in final_results[lIdx]:
            final_results[lIdx][key] = {}
        for stat in list(results[lIdx][key].values())[0].keys():
            final_results[lIdx][key][stat] = (np.mean([results[lIdx][key][run][stat] for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([results[lIdx][key][run][stat] for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))

final_results = {lIdx: {stat: {key: final_results[lIdx][key][stat] for key in final_results[lIdx]} for stat in list(final_results[lIdx].values())[0].keys()} for lIdx in final_results}
print(final_results)
pickle.dump(final_results, open(f'/home/SECONDGRAM/stats/unscaledTrainingStatsFinal.pkl', 'wb'))




results = {}
for fname in tqdm([f for f in os.listdir('/home/SECONDGRAM/stats/') if f.startswith('modelingStats_')]):
    stats = pickle.load(open(f'/home/SECONDGRAM/stats/{fname}', 'rb'))
    key = ('_').join(fname.split('_')[1:-1])
    run = int(fname.split('_')[-1].split('.')[0])
    for s in stats:
        if s not in results:
            results[s] = {}
        if key not in results[s]:
            results[s][key] = {}
        results[s][key][run] = stats[s]
          
# pickle.dump(results, open(f'/home/SECONDGRAM/stats/modelingStats.pkl', 'wb'))

final_results = {}
for key in list(results.values())[0]:
    final_results[key] = {}
    final_results[key]['Mean Train Covariance'] = (np.mean([np.mean([results['Correlation'][key][run]['Train'][d]['cov'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([np.mean([results['Correlation'][key][run]['Train'][d]['cov'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Test Covariance'] = (np.mean([np.mean([results['Correlation'][key][run]['Test'][d]['cov'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([np.mean([results['Correlation'][key][run]['Test'][d]['cov'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Train Pearson'] = (np.mean([np.mean([results['Correlation'][key][run]['Train'][d]['pearson'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([np.mean([results['Correlation'][key][run]['Train'][d]['pearson'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Test Pearson'] = (np.mean([np.mean([results['Correlation'][key][run]['Test'][d]['pearson'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([np.mean([results['Correlation'][key][run]['Test'][d]['pearson'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Train Spearman'] = (np.mean([np.mean([results['Correlation'][key][run]['Train'][d]['spearman'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([np.mean([results['Correlation'][key][run]['Train'][d]['spearman'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Test Spearman'] = (np.mean([np.mean([results['Correlation'][key][run]['Test'][d]['spearman'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([np.mean([results['Correlation'][key][run]['Test'][d]['spearman'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Train Kendall Tau'] = (np.mean([np.mean([results['Correlation'][key][run]['Train'][d]['kendalltau'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([np.mean([results['Correlation'][key][run]['Train'][d]['kendalltau'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Test Kendall Tau'] = (np.mean([np.mean([results['Correlation'][key][run]['Test'][d]['kendalltau'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([np.mean([results['Correlation'][key][run]['Test'][d]['kendalltau'] for d in range(IMAGE_DIM)]) for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Train vs. Synthetic Mean Correlation'] = (np.mean([results['Distribution'][key][run]['Train Real vs. Train Synthetic'][0] for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([results['Distribution'][key][run]['Train Real vs. Train Synthetic'][0] for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Train vs. Test Mean Correlation'] = (np.mean([results['Distribution'][key][run]['Train Real vs. Test Real'][0] for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([results['Distribution'][key][run]['Train Real vs. Test Real'][0] for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Train vs. Synthetic Covariance Difference'] = (np.mean([results['Covariate Matrix'][key][run]['Train Real vs. Train Synthetic'] for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([results['Covariate Matrix'][key][run]['Train Real vs. Train Synthetic'] for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Train vs. Test Covariance Difference'] = (np.mean([results['Covariate Matrix'][key][run]['Train Real vs. Test Real'] for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([results['Covariate Matrix'][key][run]['Train Real vs. Test Real'] for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Train Closeness'] = (np.mean([results['Closeness'][key][run]['Train']['Overall'] for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([results['Closeness'][key][run]['Train']['Overall'] for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))
    final_results[key]['Mean Test Closeness'] = (np.mean([results['Closeness'][key][run]['Test']['Overall'] for run in range(1,NUM_RUNS+1)]), 1.96 * np.std([results['Closeness'][key][run]['Test']['Overall'] for run in range(1,NUM_RUNS+1)]) / np.sqrt(NUM_RUNS))

final_results = {s: {k: final_results[k][s] for k in final_results} for s in list(final_results.values())[0].keys()}
print(final_results)
pickle.dump(final_results, open(f'/home/SECONDGRAM/stats/modelingStatsFinal.pkl', 'wb'))