import torch
import pickle
import random
import numpy as np 
from sys import argv
from config import Config
from models.ganModel import Generator

model_map = {
    'gan': 'gan_model',
}

trainData = pickle.load(open('/home/SECONDGRAM/data/trainData.pkl', 'rb'))
valData = pickle.load(open('/home/SECONDGRAM/data/valData.pkl', 'rb'))
testData = pickle.load(open('/home/SECONDGRAM/data/testData.pkl', 'rb'))

config = Config()
BATCH_SIZE = config.batch_size
EMBED_DIM = config.embed_dim
LATENT_DIM = config.latent_dim
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_DIM = len(trainData[0][1])
COND_DIM = len(trainData[0][0]) + IMAGE_DIM
NUM_RUNS = config.num_runs

def get_batch(dataset, loc, batch_size):
    data = dataset[loc:loc+batch_size]
    bs = len(data)
    condData = torch.zeros(bs, COND_DIM)
    imageData = torch.zeros(bs, IMAGE_DIM)
    for i, d in enumerate(data):
        imageData[i] = d[2] if d[2] is not None else torch.zeros(IMAGE_DIM)
        condData[i,:-IMAGE_DIM] = d[0]
        condData[i,-IMAGE_DIM:] = d[1]

    return imageData.to(DEVICE), condData.to(DEVICE)

def sample(generator, labels):
    model.eval()
    with torch.inference_mode():
        z = torch.randn(labels.size(0), LATENT_DIM, device=DEVICE)
        images = generator(z, labels)
    return images

key = 'gan'
run = int(argv[1])

random.seed(run)
torch.manual_seed(run)
np.random.seed(run)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(run)
        
model = Generator(IMAGE_DIM, LATENT_DIM, embed_dim=EMBED_DIM, condition=True, cond_dim=COND_DIM).to(DEVICE)
model.load_state_dict(torch.load(f'/home/SECONDGRAM/save/{model_map[key]}_{run}', map_location='cpu')['generator'])

generatedTrainData = []
for i in range(0, len(trainData), BATCH_SIZE):
    _, condData = get_batch(trainData, i, BATCH_SIZE)
    genData = sample(model, condData).to('cpu')
    for j, d in enumerate(trainData[i:i+BATCH_SIZE]):
        generatedTrainData.append(genData[j].clone())

generatedValData = []
for i in range(0, len(valData), BATCH_SIZE):
    _, condData = get_batch(valData, i, BATCH_SIZE)
    genData = sample(model, condData).to('cpu')
    for j, d in enumerate(valData[i:i+BATCH_SIZE]):
        generatedValData.append(genData[j].clone())

generatedTestData = []
for i in range(0, len(testData), BATCH_SIZE):
    _, condData = get_batch(testData, i, BATCH_SIZE)
    genData = sample(model, condData).to('cpu')
    for j, d in enumerate(testData[i:i+BATCH_SIZE]):
        generatedTestData.append(genData[j].clone())

pickle.dump(generatedTrainData, open(f'/home/SECONDGRAM/generations/generatedTrainData_{key}_{run}.pkl', 'wb'))
pickle.dump(generatedValData, open(f'/home/SECONDGRAM/generations/generatedValData_{key}_{run}.pkl', 'wb'))
pickle.dump(generatedTestData, open(f'/home/SECONDGRAM/generations/generatedTestData_{key}_{run}.pkl', 'wb'))