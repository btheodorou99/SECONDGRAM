import torch
import pickle
import random
import numpy as np
from sys import argv
from models.diffusionModel import AutoEncoder
from config import Config
from scipy.stats import pearsonr
from torch.nn.utils import clip_grad_norm_

trainData = pickle.load(open('/data/imageGen/data/trainData.pkl', 'rb'))
unconditionedTrainData = [(d[0], d[1], None) for d in trainData]
conditionedTrainData = [d for d in trainData if d[2] is not None]
valData = pickle.load(open('/data/imageGen/data/valData.pkl', 'rb'))
unconditionedValData = [d for d in valData if d[2] is None]
valData = [d for d in valData if d[2] is not None]
trainData = unconditionedTrainData + conditionedTrainData + unconditionedValData

config = Config()
EPOCHS = config.epochs
PATIENCE = config.patience
LR = config.lr
BATCH_SIZE = config.batch_size
NOISE_STEPS = config.noise_steps
EMBED_DIM = config.embed_dim
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_DIM = len(trainData[0][1])
COND_DIM = len(trainData[0][0]) + IMAGE_DIM
PRETRAIN_NOISE = config.pretrain_noise
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
        if d[2] is not None:
            imageData[i] = d[2]
            condData[i,:-IMAGE_DIM] = d[0]
            condData[i,-IMAGE_DIM:] = d[1]
        else:
            imageData[i] = d[1]
            condData[i,:-IMAGE_DIM] = d[0]
            randCondIndex = np.random.randint(2)
            if randCondIndex == 0:
                condData[i,-IMAGE_DIM:] = d[1]
            # elif randCondIndex == 1:
            #     condData[i,-IMAGE_DIM:] = 0
            elif randCondIndex == 1:
                condData[i,-IMAGE_DIM:] = noise_image(d[1], PRETRAIN_NOISE)

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

def sample_timesteps(n):
    return torch.randint(low=1, high=NOISE_STEPS, size=(n,)).to(DEVICE)

def noise_images(x, t):
    "Add noise to images at instant t"
    sqrt_alpha_hat = torch.sqrt(ALPHA_HAT[t])[:, None].to(DEVICE)
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - ALPHA_HAT[t])[:, None].to(DEVICE)
    Ɛ = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

def noise_image(x, t):
    "Add noise to image at instant t"
    sqrt_alpha_hat = torch.sqrt(ALPHA_HAT[t])
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - ALPHA_HAT[t])
    Ɛ = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ

run = int(argv[1])
print(f'SEED: {run}')
random.seed(run)
torch.manual_seed(run)
np.random.seed(run)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(run)

model = AutoEncoder(IMAGE_DIM, embed_dim=EMBED_DIM, condition=True, cond_dim=COND_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

patience = 0
maxCorrelation = -1
mse = torch.nn.MSELoss()
for e in range(EPOCHS):
    np.random.shuffle(trainData)
    train_losses = []
    model.train()
    for i in range(0, len(trainData), BATCH_SIZE):
        imageData, condData = get_batch(trainData, i, BATCH_SIZE)
        t = sample_timesteps(len(imageData))
        imageDataNoise, noise = noise_images(imageData, t)
        predictedNoise = model(imageDataNoise, t, condData)
        loss = mse(noise, predictedNoise)
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        norm = total_norm ** (1. / 2)
        if norm > 1.8:
            continue
        _ = clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        train_losses.append(loss.detach().cpu().item())

    val_samples = []
    val_correlations = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(valData), BATCH_SIZE):
            imageData, condData = get_batch(valData, i, BATCH_SIZE)
            sampleImages = sample(model, condData)
            for j in range(len(sampleImages)):
                val_samples.append(sampleImages[j].detach().cpu().clone())

        currTrainLoss = np.mean(train_losses)
        for d in range(IMAGE_DIM):
            val_correlations.append(pearsonr([r[2][d] for r in valData], [s[d] for s in val_samples])[0])
        currCorrelation = np.mean(val_correlations)
        print(f'Epoch: {e}; Train Loss: {currTrainLoss}; Validation Correlation: {currCorrelation}')
        if currCorrelation > maxCorrelation:
            patience = 0
            maxCorrelation = currCorrelation
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': e,
                'mode': 'train'
            }, f'/data/imageGen/save/secondgram_model_{run}')
        else:
            patience += 1
            if patience == PATIENCE:
                break