import os
import pickle
import numpy as np
from tqdm import tqdm
from config import Config

config = Config()
NUM_RUNS = config.num_runs
IMAGE_DIM = len(pickle.load(open('/data/imageGen/data/testData.pkl', 'rb'))[0][1])

LABEL_IDX = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13]
results = {}
for fname in [f for f in os.listdir('/data/imageGen/stats/') if f.startswith('trainingStats_')]:
    stats = pickle.load(open(f'/data/imageGen/stats/{fname}', 'rb'))
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
          
pickle.dump(results, open(f'/data/imageGen/stats/trainingStats.pkl', 'wb'))

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
pickle.dump(final_results, open(f'/data/imageGen/stats/trainingStatsFinal.pkl', 'wb'))





LABEL_IDX = [1, 2, 3, 4]
results = {}
for fname in [f for f in os.listdir('/data/imageGen/stats/') if f.startswith('unscaledTrainingStats_')]:
    stats = pickle.load(open(f'/data/imageGen/stats/{fname}', 'rb'))
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
          
pickle.dump(results, open(f'/data/imageGen/stats/unscaledTrainingStats.pkl', 'wb'))

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
pickle.dump(final_results, open(f'/data/imageGen/stats/unscaledTrainingStatsFinal.pkl', 'wb'))