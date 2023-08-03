import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from models.baseModel import AutoEncoder
from config import Config
from scipy.stats import pearsonr

trainData = pickle.load(open('/data/theodoroubp/imageGen/data/trainData.pkl', 'rb'))
pretrainData = [d for d in trainData]
trainData = [d for d in trainData if d[2] is not None]
valData = pickle.load(open('/data/theodoroubp/imageGen/data/valData.pkl', 'rb'))
prevalData = [d for d in valData]
valData = [d for d in valData if d[2] is not None]

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
NUM_RUNS = config.num_runs

BETA_START = config.beta_start
BETA_END = config.beta_end
BETA = torch.linspace(BETA_START, BETA_END, NOISE_STEPS)
ALPHA = 1 - BETA
ALPHA_HAT = torch.cumprod(ALPHA, dim=0)

def get_prebatch(dataset, loc, batch_size):
    data = dataset[loc:loc+batch_size]
    bs = len(data)
    imageData = torch.zeros(bs, IMAGE_DIM)
    for i, d in enumerate(data):
        imageData[i] = d[1]

    return imageData.to(DEVICE)

def get_batch(dataset, loc, batch_size):
    data = dataset[loc:loc+batch_size]
    bs = len(data)
    condData = torch.zeros(bs, COND_DIM)
    imageData = torch.zeros(bs, IMAGE_DIM)
    for i, d in enumerate(data):
        imageData[i] = d[2]
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

for run in tqdm(range(NUM_RUNS), desc='Pretrained Runs'):
    random.seed(run)
    torch.manual_seed(run)
    np.random.seed(run)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run)

    model = AutoEncoder(IMAGE_DIM, embed_dim=EMBED_DIM, condition=True, cond_dim=COND_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    patience = 0
    global_loss = 1e10
    mse = torch.nn.MSELoss()
    for e in range(EPOCHS):
        np.random.shuffle(pretrainData)
        train_losses = []
        model.train()
        for i in tqdm(range(0, len(pretrainData), BATCH_SIZE), leave=False):
            imageData = get_prebatch(pretrainData, i, BATCH_SIZE)
            t = sample_timesteps(len(imageData))
            imageDataNoise, noise = noise_images(imageData, t)
            predictedNoise = model(imageDataNoise, t)
            loss = mse(noise, predictedNoise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())

        val_losses = []
        model.eval()
        with torch.inference_mode():
            for i in tqdm(range(0, len(prevalData), BATCH_SIZE), leave=False):
                imageData = get_prebatch(prevalData, i, BATCH_SIZE)
                t = sample_timesteps(len(imageData))
                imageDataNoise, noise = noise_images(imageData, t)
                predictedNoise = model(imageDataNoise, t)
                loss = mse(noise, predictedNoise)
                val_losses.append(loss.detach().cpu().item())

            currTrainLoss = np.mean(train_losses)
            currValLoss = np.mean(val_losses)
            print(f'Epoch: {e}; Train Loss: {currTrainLoss}; Validation Loss: {currValLoss}')
            if currValLoss < global_loss:
                patience = 0
                global_loss = currValLoss
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': e,
                    'mode': 'pretrain'
                }, f'/data/theodoroubp/imageGen/save/pretrained_model_{run}')
            else:
                patience += 1
                if patience == PATIENCE:
                    break

    model.load_state_dict(torch.load(f'/data/theodoroubp/imageGen/save/pretrained_model_{run}', map_location='cpu')['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    patience = 0
    maxCorrelation = -1
    mse = torch.nn.MSELoss()
    for e in range(EPOCHS):
        np.random.shuffle(trainData)
        train_losses = []
        model.train()
        for i in tqdm(range(0, len(trainData), BATCH_SIZE), leave=False):
            imageData, condData = get_batch(trainData, i, BATCH_SIZE)
            t = sample_timesteps(len(imageData))
            imageDataNoise, noise = noise_images(imageData, t)
            predictedNoise = model(imageDataNoise, t, condData)
            loss = mse(noise, predictedNoise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())

        val_samples = []
        val_correlations = []
        model.eval()
        with torch.inference_mode():
            for i in tqdm(range(0, len(valData), BATCH_SIZE), leave=False):
                imageData, condData = get_batch(valData, i, BATCH_SIZE)
                sampleImages = sample(model, condData)
                for j in range(len(sampleImages)):
                    val_samples.append(sampleImages[j].detach().cpu().clone())

            currTrainLoss = np.mean(train_losses)
            for d in tqdm(range(IMAGE_DIM), leave=False, desc='Correlation'):
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
                }, f'/data/theodoroubp/imageGen/save/pretrained_model_{run}')
            else:
                patience += 1
                if patience == PATIENCE:
                    break