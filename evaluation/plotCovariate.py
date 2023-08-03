import matplotlib.pyplot as plt
from config import Config
import numpy as np
import pickle

config = Config()
IMAGE_DIM = pickle.load(open('/data/theodoroubp/imageGen/data/testData.pkl', 'rb'))[0][1].shape[0]
NUM_RUNS = config.num_runs
covariates = pickle.load(open(f'/data/theodoroubp/imageGen/stats/modelingStats.pkl', 'rb'))['Covariate Matrix']
for key in covariates:
    pairs = {}
    for pair in covariates[key][0]['Train Synthetic']:
        pairs[pair] = np.mean([covariates[key][run]['Train Synthetic'][pair]['pearson'] for run in range(NUM_RUNS)])

    plt.figure()
    plt.imshow(np.array([[pairs[(d, d2) if d <= d2 else (d2,d)] for d2 in range(IMAGE_DIM)] for d in range(IMAGE_DIM)]))
    plt.colorbar()
    plt.title(f'{key} Covariate Matrix')
    plt.savefig(f'/data/theodoroubp/imageGen/stats/{key}_covariate_matrix.png')



key = 'base'
pairs = {}
for pair in covariates[key][0]['Train Synthetic']:
    pairs[pair] = np.mean([covariates[key][run]['Train Real'][pair]['pearson'] for run in range(NUM_RUNS)])

plt.figure()
plt.imshow(np.array([[pairs[(d, d2) if d <= d2 else (d2,d)] for d2 in range(IMAGE_DIM)] for d in range(IMAGE_DIM)]))
plt.colorbar()
plt.title(f'{key} Covariate Matrix')
plt.savefig(f'/data/theodoroubp/imageGen/stats/real_covariate_matrix.png')