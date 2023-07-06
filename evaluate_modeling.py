import math
import pickle
import numpy as np 
from tqdm import tqdm
from scipy.stats import pearsonr

# POSSIBLE KEYS
# base
# bnn
# pretrained
# pretrained_demo
# pretrained_self
# proposed
# gan
# pretrained_gan

key = 'base'
generatedTrainData = pickle.load(open(f'/data/theodoroubp/imageGen/generatedTrainData_{key}.pkl', 'rb'))
generatedTestData = pickle.load(open(f'/data/theodoroubp/imageGen/generatedTestData_{key}.pkl', 'rb'))
imputedTrainData = [(d[2], d[3]) for d in generatedTrainData if d[2] is not None]
allTrainReal = [d[2] for d in generatedTrainData if d[2] is not None]
allTrainSynthetic = [d[3] for d in generatedTrainData if d[2] is not None]
imputedTestData = [(d[2], d[3]) for d in generatedTestData if d[2] is not None]
allTestReal = [d[2] for d in generatedTestData if d[2] is not None]
allTestSynthetic = [d[3] for d in generatedTestData if d[2] is not None]

IMAGE_DIM = len(allTrainReal[0])

pearsonStats = {'Train': {}, 'Test': {}}
for d in tqdm(range(IMAGE_DIM), leave=False, desc='Pearson'):
    res = pearsonr([r[d] for r in allTrainReal], [s[d] for s in allTrainSynthetic])
    pearsonStats['Train'][d] = res
    res = pearsonr([r[d] for r in allTestReal], [s[d] for s in allTestSynthetic])
    pearsonStats['Test'][d] = res

distributionStats = {}
for label, data in tqdm([('Train Real', allTrainReal), ('Train Synthetic', allTrainSynthetic), ('Test Real', allTestReal), ('Test Synthetic', allTestSynthetic)], leave=False, desc='Distribution'):
    distributionStats[label] = {}
    for d in tqdm(range(IMAGE_DIM), leave=False, desc=label):
        distributionStats[label][d] = {}
        distributionStats[label][d]['mean'] = np.mean([r[d] for r in data])
        distributionStats[label][d]['std'] = np.std([r[d] for r in data])
        distributionStats[label][d]['min'] = np.min([r[d] for r in data])
        distributionStats[label][d]['max'] = np.max([r[d] for r in data])

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
    'Pearson': pearsonStats,
    'Distribution': distributionStats,
    'Closeness': closenessStats
}

print('Mean Train Pearson: {}'.format(np.mean([p[0] for p in pearsonStats['Train'].values()])))
print('Mean Test Pearson: {}'.format(np.mean([p[0] for p in pearsonStats['Test'].values()])))
print('Mean Train Closeness: {}'.format(closenessStats['Train']['Overall']))
print('Mean Test Closeness: {}'.format(closenessStats['Test']['Overall']))

pickle.dump(stats, open(f'/data/theodoroubp/imageGen/stats_{key}.pkl', 'wb'))
