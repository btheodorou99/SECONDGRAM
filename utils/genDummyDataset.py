import torch
import pickle
import random
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Number of data points
num_data_points = 41706  # Total number of entries
num_second_features = 4997  # Number of entries with second features

# Generate static features: 13 floats with 10% probability of being 1, else 0
staticFeatures = {e: torch.FloatTensor([1.0 if random.random() < 0.1 else 0.0 for _ in range(13)]) for e in range(num_data_points)}

# Generate first features: random continuous values from -1 to 1 of dimension 769
firstFeatures = {e: torch.FloatTensor(769).uniform_(-1, 1) for e in range(num_data_points)}

# Select indices for second features
secondFeatureIndices = set(random.sample(range(num_data_points), num_second_features))

# Generate second features: perturbed first features with random noise added, constrained between -1 and 1
secondFeatures = {
    e: torch.clamp(firstFeatures[e] + torch.randn_like(firstFeatures[e]) * 0.1, -1, 1)
    for e in secondFeatureIndices
}

# Create data list
data = [
    (torch.FloatTensor(staticFeatures[e]),
     torch.FloatTensor(firstFeatures[e]),
     torch.FloatTensor(secondFeatures[e]) if e in secondFeatures else None)
    for e in range(num_data_points)
]

trainData, testData = train_test_split(data, test_size=0.2)
trainData, valData = train_test_split(trainData, test_size=0.1)
pickle.dump(trainData, open('/home/SECONDGRAM/data/trainData.pkl', 'wb'))
pickle.dump(valData, open('/home/SECONDGRAM/data/valData.pkl', 'wb'))
pickle.dump(testData, open('/home/SECONDGRAM/data/testData.pkl', 'wb'))

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
#         Number of Second Images: 3605

# Validation
#         Dataset Size: 3337
#         Number of Second Images: 378

# Test
#         Dataset Size: 8342
#         Number of Second Images: 1014

# Demographic Dimensionality: 14
# Image Dimensionality: 769