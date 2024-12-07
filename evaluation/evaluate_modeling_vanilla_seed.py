import math
import pickle
import numpy as np 
from sys import argv
from tqdm import tqdm
from config import Config
from scipy.stats import pearsonr, spearmanr, kendalltau

key = 'vanilla'

config = Config()
NUM_RUNS = config.num_runs
realTrainData = pickle.load(open('/home/SECONDGRAM/data/trainData.pkl', 'rb'))
realTestData = pickle.load(open('/home/SECONDGRAM/data/testData.pkl', 'rb'))
IMAGE_DIM = len(realTrainData[0][1])

run = int(argv[1])
generatedTrainData = pickle.load(open(f'/home/SECONDGRAM/generations/generatedTrainData_{key}_{run}.pkl', 'rb'))
generatedTrainData = [(r[0], r[1], r[2], g) for r, g in zip(realTrainData, generatedTrainData)]
generatedTestData = pickle.load(open(f'/home/SECONDGRAM/generations/generatedTestData_{key}_{run}.pkl', 'rb'))
generatedTestData = [(r[0], r[1], r[2], g) for r, g in zip(realTestData, generatedTestData)]
imputedTrainData = [(d[2], d[3]) for d in generatedTrainData if d[2] is not None]
allTrainReal = [d[2] for d in generatedTrainData if d[2] is not None]
allTrainSynthetic = [d[3] for d in generatedTrainData if d[2] is not None]
imputedTestData = [(d[2], d[3]) for d in generatedTestData if d[2] is not None]
allTestReal = [d[2] for d in generatedTestData if d[2] is not None]
allTestSynthetic = [d[3] for d in generatedTestData if d[2] is not None]

correlationStats = {'Train': {}, 'Test': {}}
for d in tqdm(range(IMAGE_DIM), leave=False, desc='Correlation'):
    correlationStats['Train'][d] = {}
    correlationStats['Test'][d] = {}
    correlationStats['Train'][d]['cov'] = np.cov([r[d] for r in allTrainReal], [s[d] for s in allTrainSynthetic])[0][1]
    correlationStats['Test'][d]['cov'] = np.cov([r[d] for r in allTestReal], [s[d] for s in allTestSynthetic])[0][1]
    correlationStats['Train'][d]['pearson'] = pearsonr([r[d] for r in allTrainReal], [s[d] for s in allTrainSynthetic])[0]
    correlationStats['Test'][d]['pearson'] = pearsonr([r[d] for r in allTestReal], [s[d] for s in allTestSynthetic])[0]
    correlationStats['Train'][d]['spearman'] = spearmanr([r[d] for r in allTrainReal], [s[d] for s in allTrainSynthetic])[0]
    correlationStats['Test'][d]['spearman'] = spearmanr([r[d] for r in allTestReal], [s[d] for s in allTestSynthetic])[0]
    correlationStats['Train'][d]['kendalltau'] = kendalltau([r[d] for r in allTrainReal], [s[d] for s in allTrainSynthetic])[0]
    correlationStats['Test'][d]['kendalltau'] = kendalltau([r[d] for r in allTestReal], [s[d] for s in allTestSynthetic])[0]

distributionStats = {}
for label, data in tqdm([('Train Real', allTrainReal), ('Train Synthetic', allTrainSynthetic), ('Test Real', allTestReal), ('Test Synthetic', allTestSynthetic)], leave=False, desc='Distribution'):
    distributionStats[label] = {}
    for d in tqdm(range(IMAGE_DIM), leave=False, desc=label):
        distributionStats[label][d] = {}
        distributionStats[label][d]['mean'] = np.mean([r[d] for r in data])
        distributionStats[label][d]['std'] = np.std([r[d] for r in data])
        distributionStats[label][d]['min'] = np.min([r[d] for r in data])
        distributionStats[label][d]['max'] = np.max([r[d] for r in data])

distributionStats['Train Real vs. Train Synthetic'] = pearsonr([distributionStats['Train Real'][d]['mean'] for d in range(IMAGE_DIM)], [distributionStats['Train Synthetic'][d]['mean'] for d in range(IMAGE_DIM)])
distributionStats['Train Real vs. Test Real'] = pearsonr([distributionStats['Train Real'][d]['mean'] for d in range(IMAGE_DIM)], [distributionStats['Test Real'][d]['mean'] for d in range(IMAGE_DIM)])

matrixStats = {}
for label, data in tqdm([('Train Real', allTrainReal), ('Train Synthetic', allTrainSynthetic), ('Test Real', allTestReal), ('Test Synthetic', allTestSynthetic)], leave=False, desc='Covariance'):
    matrixStats[label] = {}
    for d in tqdm(range(IMAGE_DIM), leave=False, desc=label):
        for d2 in range(d, IMAGE_DIM):
            matrixStats[label][(d, d2)] = {}
            matrixStats[label][(d, d2)]['pearson'] = pearsonr([r[d] for r in data], [r[d2] for r in data])[0]

matrixStats['Train Real vs. Train Synthetic'] = np.mean([np.abs(matrixStats['Train Real'][(d, d2)]['pearson'] - matrixStats['Train Synthetic'][(d, d2)]['pearson']) for d in range(IMAGE_DIM) for d2 in range(d, IMAGE_DIM)])
matrixStats['Train Real vs. Test Real'] = np.mean([np.abs(matrixStats['Train Real'][(d, d2)]['pearson'] - matrixStats['Test Real'][(d, d2)]['pearson']) for d in range(IMAGE_DIM) for d2 in range(d, IMAGE_DIM)])

closenessStats = {}
for dataType in tqdm(['Train', 'Test'], leave=False, desc='Closeness'):
    if dataType == 'Train':
        realData = allTrainReal
        syntheticData = allTrainSynthetic
    else:
        realData = allTestReal
        syntheticData = allTestSynthetic

    closenessStats[dataType] = {}
    for d in tqdm(range(IMAGE_DIM), leave=False, desc=dataType):
        closenessStats[dataType][d] = {}
        closenessStats[dataType][d]['mae'] = np.mean([np.abs(r[d] - s[d]) for r, s in zip(realData, syntheticData)])
        closenessStats[dataType][d]['mse'] = np.mean([(r[d] - s[d])**2 for r, s in zip(realData, syntheticData)])
    closenessStats[dataType]['Overall'] = np.mean([math.dist(r, s) for r, s in zip(realData, syntheticData)])

stats = {
    'Correlation': correlationStats,
    'Distribution': distributionStats,
    'Covariate Matrix': matrixStats,
    'Closeness': closenessStats
}

print('Mean Train Covariance: {}'.format(np.mean([correlationStats['Train'][d]['cov'] for d in range(IMAGE_DIM)])))
print('Mean Test Covariance: {}'.format(np.mean([correlationStats['Test'][d]['cov'] for d in range(IMAGE_DIM)])))
print('Mean Train Pearson: {}'.format(np.mean([correlationStats['Train'][d]['pearson'] for d in range(IMAGE_DIM)])))
print('Mean Test Pearson: {}'.format(np.mean([correlationStats['Test'][d]['pearson'] for d in range(IMAGE_DIM)])))
print('Mean Train Spearman: {}'.format(np.mean([correlationStats['Train'][d]['spearman'] for d in range(IMAGE_DIM)])))
print('Mean Test Spearman: {}'.format(np.mean([correlationStats['Test'][d]['spearman'] for d in range(IMAGE_DIM)])))
print('Mean Train Kendall Tau: {}'.format(np.mean([correlationStats['Train'][d]['kendalltau'] for d in range(IMAGE_DIM)])))
print('Mean Test Kendall Tau: {}'.format(np.mean([correlationStats['Test'][d]['kendalltau'] for d in range(IMAGE_DIM)])))

print('Train vs. Synthetic Mean Correlation: {}'.format(distributionStats['Train Real vs. Train Synthetic']))
print('Train vs. Test Mean Correlation: {}'.format(distributionStats['Train Real vs. Test Real']))

print('Mean Train vs. Synthetic Covariance Difference: {}'.format(matrixStats['Train Real vs. Train Synthetic']))

print('Mean Train Closeness: {}'.format(closenessStats['Train']['Overall']))
print('Mean Test Closeness: {}'.format(closenessStats['Test']['Overall']))

pickle.dump(stats, open(f'/home/SECONDGRAM/stats/modelingStats_{key}_{run}.pkl', 'wb'))