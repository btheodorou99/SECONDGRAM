import torch
import pickle
import numpy as np 
from tqdm import tqdm
from config import Config
from ganModel import Generator

# POSSIBLE KEYS
# gan
# pretrained_gan
# pretrained_gan_demo

model_map = {
    'gan': 'gan_model',
    'pretrained_gan': 'pretrained_gan_model',
    'pretrained_gan_demo': 'pretrained_gan_model_demo',
}

key = 'gan'

trainData = pickle.load(open('/data/theodoroubp/imageGen/trainData.pkl', 'rb'))
valData = pickle.load(open('/data/theodoroubp/imageGen/valData.pkl', 'rb'))
testData = pickle.load(open('/data/theodoroubp/imageGen/testData.pkl', 'rb'))

config = Config()
BATCH_SIZE = config.batch_size
EMBED_DIM = config.embed_dim
LATENT_DIM = config.latent_dim
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_DIM = len(trainData[0][1])
COND_DIM = len(trainData[0][0]) + IMAGE_DIM

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
        z = torch.randn(labels.size(0), LATENT_DIM)
        images = generator(z, labels)
    return images

model = Generator(IMAGE_DIM, LATENT_DIM, embed_dim=EMBED_DIM, condition=True, cond_dim=COND_DIM).to(DEVICE)
model.load_state_dict(torch.load(f'/data/theodoroubp/imageGen/save/{model_map[key]}', map_location='cpu')['generator'])

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
