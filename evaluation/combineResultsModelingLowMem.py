import os
import pickle
import numpy as np
from tqdm import tqdm
from config import Config

config = Config()
NUM_RUNS = config.num_runs
IMAGE_DIM = len(pickle.load(open('/home/SECONDGRAM/data/testData.pkl', 'rb'))[0][1])

allFiles = [f for f in os.listdir('/home/SECONDGRAM/stats/') if f.startswith('modelingStats_')]
allKeys = set([('_').join(f.split('_')[1:-1]) for f in allFiles])
final_results = {}
for key in tqdm(allKeys):
    results = {}
    for run in tqdm(range(1,NUM_RUNS+1), leave=False):
        fname = f'modelingStats_{key}_{run}.pkl'
        stats = pickle.load(open(f'/home/SECONDGRAM/stats/{fname}', 'rb'))
        for s in stats:
            if s not in results:
                results[s] = {}
            if key not in results[s]:
                results[s][key] = {}
            results[s][key][run] = stats[s]
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