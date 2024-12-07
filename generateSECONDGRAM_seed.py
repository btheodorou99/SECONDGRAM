import torch
import random
import pickle
import numpy as np 
from sys import argv
from config import Config
from models.diffusionModel import AutoEncoder

model_map = {
    'vanilla': 'vanilla_model',
    'pretrained': 'pretrained_model',
    'secondgram_noCombined': 'secondgramNoCombined_model',
    'secondgram_noGrad': 'secondgramNoGrad_model',
    'secondgram': 'secongram_model',
}

trainData = pickle.load(open('/home/SECONDGRAM/data/trainData.pkl', 'rb'))
valData = pickle.load(open('/home/SECONDGRAM/data/valData.pkl', 'rb'))
testData = pickle.load(open('/home/SECONDGRAM/data/testData.pkl', 'rb'))

config = Config()
BATCH_SIZE = config.batch_size
NOISE_STEPS = config.noise_steps
EMBED_DIM = config.embed_dim
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_DIM = len(trainData[0][1])
COND_DIM = len(trainData[0][0]) + IMAGE_DIM
NUM_RUNS = config.num_runs

BETA_START = config.beta_start
BETA_END = config.beta_end
BETA = torch.linspace(BETA_START, BETA_END, NOISE_STEPS)
ALPHA = 1 - BETA
ALPHA_HAT = torch.cumprod(ALPHA, dim=0)

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

def sample(model, labels):
    n = len(labels)
    model.eval()
    with torch.inference_mode():
        x = torch.randn((n, IMAGE_DIM)).to(DEVICE)
        for timestep in reversed(range(1, NOISE_STEPS)):
            t = (torch.ones(n) * timestep).long().to(DEVICE)
            predicted_noise = model(x, t, labels)
            alpha = ALPHA[t][:, None].to(DEVICE)
            alpha_hat = ALPHA_HAT[t][:, None].to(DEVICE)
            beta = BETA[t][:, None].to(DEVICE)
            if i > 1:
                noise = torch.randn_like(x, device=DEVICE)
            else:
                noise = torch.zeros_like(x, device=DEVICE)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    x = x.clamp(-1,1)
    return x

key = 'secondgram'

run = int(argv[1])
random.seed(run)
torch.manual_seed(run)
np.random.seed(run)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(run)

model = AutoEncoder(IMAGE_DIM, embed_dim=EMBED_DIM, condition=True, cond_dim=COND_DIM).to(DEVICE)
model.load_state_dict(torch.load(f'/home/SECONDGRAM/save/{model_map[key]}_{run}', map_location='cpu')['model'])

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