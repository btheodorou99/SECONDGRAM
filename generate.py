import torch
import pickle
import numpy as np 
from tqdm import tqdm
from config import Config

# POSSIBLE KEYS
# base
# bnn
# pretrained
# pretrained_demo
# pretrained_self
# proposed

model_map = {
    'base': 'base_model',
    'bnn': 'bnn_model',
    'pretrained': 'pretrained_model',
    'pretrained_demo': 'pretrained_model_demo',
    'pretrained_self': 'pretrained_model_self',
    'proposed': 'proposed_model'
}


key = 'base'

if key in ['bnn', 'proposed']:
    from bnnModel import AutoEncoder
else:
    from baseModel import AutoEncoder

trainData = pickle.load(open('/data/theodoroubp/imageGen/trainData.pkl', 'rb'))
valData = pickle.load(open('/data/theodoroubp/imageGen/valData.pkl', 'rb'))
testData = pickle.load(open('/data/theodoroubp/imageGen/testData.pkl', 'rb'))

config = Config()
BATCH_SIZE = config.batch_size
NOISE_STEPS = config.noise_steps
EMBED_DIM = config.embed_dim
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_DIM = len(trainData[0][1])
COND_DIM = len(trainData[0][0]) + IMAGE_DIM

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
        for timestep in tqdm(reversed(range(1, NOISE_STEPS)), leave=False):
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
    x = (x.clamp(-1, 1) + 1) / 2
    return x

model = AutoEncoder(IMAGE_DIM, embed_dim=EMBED_DIM, condition=True, cond_dim=COND_DIM).to(DEVICE)
model.load_state_dict(torch.load(f'/data/theodoroubp/imageGen/save/{model_map[key]}', map_location='cpu')['model'])

generatedTrainData = []
for i in tqdm(range(0, len(trainData), BATCH_SIZE), leave=False, desc='Train'):
    _, condData = get_batch(trainData, i, BATCH_SIZE)
    genData = sample(model, condData).to('cpu')
    for j, d in enumerate(trainData[i:i+BATCH_SIZE]):
        generatedTrainData.append((d[0], d[1], d[2], genData[j]))

generatedValData = []
for i in tqdm(range(0, len(valData), BATCH_SIZE), leave=False, desc='Val'):
    _, condData = get_batch(valData, i, BATCH_SIZE)
    genData = sample(model, condData).to('cpu')
    for j, d in enumerate(valData[i:i+BATCH_SIZE]):
        generatedValData.append((d[0], d[1], d[2], genData[j]))

generatedTestData = []
for i in tqdm(range(0, len(testData), BATCH_SIZE), leave=False, desc='Test'):
    _, condData = get_batch(testData, i, BATCH_SIZE)
    genData = sample(model, condData).to('cpu')
    for j, d in enumerate(testData[i:i+BATCH_SIZE]):
        generatedTestData.append((d[0], d[1], d[2], genData[j]))

pickle.dump(generatedTrainData, open(f'/data/theodoroubp/imageGen/generatedTrainData_{key}.pkl', 'wb'))
pickle.dump(generatedValData, open(f'/data/theodoroubp/imageGen/generatedValData_{key}.pkl', 'wb'))
pickle.dump(generatedTestData, open(f'/data/theodoroubp/imageGen/generatedTestData_{key}.pkl', 'wb'))
