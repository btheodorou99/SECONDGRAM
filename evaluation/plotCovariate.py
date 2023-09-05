import matplotlib.pyplot as plt
from config import Config
import numpy as np
import pickle
import os

config = Config()
IMAGE_DIM = pickle.load(open('/data/imageGen/data/testData.pkl', 'rb'))[0][1].shape[0]
NUM_RUNS = config.num_runs

results = {}
for fname in [f for f in os.listdir('/data/imageGen/stats/') if f.startswith('modelingStats_')]:
    stats = pickle.load(open(f'/data/imageGen/stats/{fname}', 'rb'))
    key = ('_').join(fname.split('_')[1:-1])
    run = int(fname.split('_')[-1].split('.')[0])
    for s in stats:
        if s not in results:
            results[s] = {}
        if key not in results[s]:
            results[s][key] = {}
        results[s][key][run] = stats[s]

covariates = results['Covariate Matrix']
# covariates = pickle.load(open(f'/data/imageGen/stats/modelingStats.pkl', 'rb'))['Covariate Matrix']

for key in covariates:
    pairs = {}
    for pair in covariates[key][1]['Train Synthetic']:
        pairs[pair] = np.mean([covariates[key][run]['Train Synthetic'][pair]['pearson'] for run in range(1,NUM_RUNS+1)])

    plt.figure()
    plt.imshow(np.array([[pairs[(d, d2) if d <= d2 else (d2,d)] for d2 in range(IMAGE_DIM)] for d in range(IMAGE_DIM)]))
    plt.colorbar()
    plt.title(f'{key} Covariate Matrix')
    plt.savefig(f'/data/imageGen/stats/{key}_covariate_matrix.png')



key = 'base'
pairs = {}
for pair in covariates[key][1]['Train Synthetic']:
    pairs[pair] = np.mean([covariates[key][run]['Train Real'][pair]['pearson'] for run in range(1,NUM_RUNS+1)])

plt.figure()
plt.imshow(np.array([[pairs[(d, d2) if d <= d2 else (d2,d)] for d2 in range(IMAGE_DIM)] for d in range(IMAGE_DIM)]))
plt.colorbar()
plt.title(f'{key} Covariate Matrix')
plt.savefig(f'/data/imageGen/stats/real_covariate_matrix.png')