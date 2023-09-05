import os
import torch
import pickle
import random
import numpy as np 
import torch.nn as nn
from sys import argv
from tqdm import tqdm
from config import Config
from sklearn import metrics

# Train models on:
# 1. All first images
# 2. All paired images (real where present, imputed where not)
# 3. All paired images (imputed only)
# 4. Subset (where there is a second image) first images
# 5. Subset paired images (real second)
# 6. Subset paired images (imputed second)

key = 'pretrained'

testData = pickle.load(open('/data/imageGen/data/testData.pkl', 'rb'))
LABEL_IDX = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13]
EPOCHS = 1000
PATIENCE = 5
LR = 1e-3
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
config = Config()
NUM_RUNS = config.num_runs

class DownstreamModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=256, output_dim=1, binary=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, output_dim)
        self.binary = binary

    def forward(self, input):
        output = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(input)))))
        if self.binary:
            output = torch.sigmoid(output)
        return output


def get_batch(dataset, loc, batch_size):
    data = dataset[loc:loc+batch_size]
    bs = len(data)
    batch_input = torch.zeros(bs, len(data[0][0]))
    batch_labels = torch.zeros(bs, 1)
    for i, d in enumerate(data):
        batch_input[i] = d[0]
        batch_labels[i] = d[1]

    return batch_input.to(DEVICE), batch_labels.to(DEVICE)

def train_model(model, label, trainDataset, valDataset, binary=False):
    print(f'\n{label}')
    patience = 0
    global_loss = 1e10
    loss_fn = torch.nn.MSELoss() if not binary else torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for e in tqdm(range(EPOCHS), leave=False):
        np.random.shuffle(trainDataset)
        train_losses = []
        model.train()
        for i in tqdm(range(0, len(trainDataset), BATCH_SIZE), leave=False):
            batch_input, batch_labels = get_batch(trainDataset, i, BATCH_SIZE)
            batch_preds = model(batch_input)
            loss = loss_fn(batch_preds, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())

        val_losses = []
        model.eval()
        with torch.inference_mode():
            for i in tqdm(range(0, len(valDataset), BATCH_SIZE), leave=False):
                batch_input, batch_labels = get_batch(valDataset, i, BATCH_SIZE)
                batch_preds = model(batch_input)
                loss = loss_fn(batch_preds, batch_labels)
                val_losses.append(loss.detach().cpu().item())

            currTrainLoss = np.mean(train_losses)
            currValLoss = np.mean(val_losses)
            # print(f'\tEpoch: {e}; Train Loss: {currTrainLoss}; Validation Loss: {currValLoss}')
            if currValLoss < global_loss:
                patience = 0
                global_loss = currValLoss
                torch.save(model.state_dict(), f'/data/imageGen/save/downstream_model_{label}')
            else:
                patience += 1
                if patience == PATIENCE:
                    break

    model.load_state_dict(torch.load(f'/data/imageGen/save/downstream_model_{label}', map_location='cpu'))
    return model

def test_model(model, testDataset, binary=False):
    preds = []
    labels = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(testDataset), BATCH_SIZE), leave=False):
            batch_input, batch_labels = get_batch(testDataset, i, BATCH_SIZE)
            batch_preds = model(batch_input).relu()
            preds += batch_preds.squeeze(-1).cpu().tolist()
            labels += batch_labels.squeeze(-1).cpu().tolist()

    if binary:
        rounded_preds = [1 if p >= 0.5 else 0 for p in preds]
        stats = {
            'Accuracy': metrics.accuracy_score(labels, rounded_preds),
            'Precision': metrics.precision_score(labels, rounded_preds, zero_division=0),
            'Recall': metrics.recall_score(labels, rounded_preds),
            'F1': metrics.f1_score(labels, rounded_preds),
            'AUC': metrics.roc_auc_score(labels, preds)
        }
    else:
        stats = {
            'MAE': metrics.mean_absolute_error(labels, preds),
            'MSE': metrics.mean_squared_error(labels, preds)
        }

    print(stats)
    return stats

realTrainData = pickle.load(open('/data/imageGen/data/trainData.pkl', 'rb'))
realTestData = pickle.load(open('/data/imageGen/data/testData.pkl', 'rb'))

run = int(argv[1])

generatedTrainData = pickle.load(open(f'/data/imageGen/generations/generatedTrainData_{key}_{run}.pkl', 'rb'))
generatedTrainData = [(r[0], r[1], r[2], g) for r, g in zip(realTrainData, generatedTrainData)]
generatedValData = pickle.load(open(f'/data/imageGen/generations/generatedValData_{key}_{run}.pkl', 'rb'))
generatedValData = [(r[0], r[1], r[2], g) for r, g in zip(realTrainData, generatedValData)]

for l_idx in LABEL_IDX:
    random.seed(run)
    torch.manual_seed(run)
    np.random.seed(run)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run)
        
    print(f'\n\nLABEL: {l_idx}\n')
    train1 = [(d[1], d[0][l_idx]) for d in generatedTrainData]
    train2 = [(torch.cat((d[1], d[2] if d[2] is not None else d[3])), d[0][l_idx]) for d in generatedTrainData]
    train3 = [(torch.cat((d[1], d[3])), d[0][l_idx]) for d in generatedTrainData]
    train4 = [(d[1], d[0][l_idx]) for d in generatedTrainData if d[2] is not None]
    train5 = [(torch.cat((d[1], d[2])), d[0][l_idx]) for d in generatedTrainData if d[2] is not None]
    train6 = [(torch.cat((d[1], d[3])), d[0][l_idx]) for d in generatedTrainData if d[2] is not None]
    val1 = [(d[1], d[0][l_idx]) for d in generatedValData]
    val2 = [(torch.cat((d[1], d[2] if d[2] is not None else d[3])), d[0][l_idx]) for d in generatedValData]
    val3 = [(torch.cat((d[1], d[3])), d[0][l_idx]) for d in generatedValData]
    val4 = [(d[1], d[0][l_idx]) for d in generatedValData if d[2] is not None]
    val5 = [(torch.cat((d[1], d[2])), d[0][l_idx]) for d in generatedValData if d[2] is not None]
    val6 = [(torch.cat((d[1], d[3])), d[0][l_idx]) for d in generatedValData if d[2] is not None]
    testDataSingle = [(d[1], d[0][l_idx]) for d in testData if d[2] is not None]
    testDataPaired = [(torch.cat((d[1], d[2])), d[0][l_idx]) for d in testData if d[2] is not None]
    binary = l_idx in [0, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    stats = {}

    model = DownstreamModel(len(train1[0][0]), binary=binary).to(DEVICE)
    model = train_model(model, f'AllFirst_{l_idx}_{key}_{run}', train1, val1, binary=binary)
    stats['AllFirst'] = test_model(model, testDataSingle, binary=binary)

    model = DownstreamModel(len(train2[0][0]), binary=binary).to(DEVICE)
    model = train_model(model, f'AllPairedImputed_{l_idx}_{key}_{run}', train2, val2, binary=binary)
    stats['AllPairedImputed'] = test_model(model, testDataPaired, binary=binary)

    model = DownstreamModel(len(train3[0][0]), binary=binary).to(DEVICE)
    model = train_model(model, f'AllPairedSynthetic_{l_idx}_{key}_{run}', train3, val3, binary=binary)
    stats['AllPairedSynthetic'] = test_model(model, testDataPaired, binary=binary)

    model = DownstreamModel(len(train4[0][0]), binary=binary).to(DEVICE)
    model = train_model(model, f'SubsetFirst_{l_idx}_{key}_{run}', train4, val4, binary=binary)
    stats['SubsetFirst'] = test_model(model, testDataSingle, binary=binary)

    model = DownstreamModel(len(train5[0][0]), binary=binary).to(DEVICE)
    model = train_model(model, f'SubsetPairedReal_{l_idx}_{key}_{run}', train5, val5, binary=binary)
    stats['SubsetPairedReal'] = test_model(model, testDataPaired, binary=binary)

    model = DownstreamModel(len(train6[0][0]), binary=binary).to(DEVICE)
    model = train_model(model, f'SubsetPairedSynthetic_{l_idx}_{key}_{run}', train6, val6, binary=binary)
    stats['SubsetPairedSynthetic'] = test_model(model, testDataPaired, binary=binary)

    os.remove(f'/data/imageGen/save/downstream_model_AllFirst_{l_idx}_{key}_{run}')
    os.remove(f'/data/imageGen/save/downstream_model_AllPairedImputed_{l_idx}_{key}_{run}')
    os.remove(f'/data/imageGen/save/downstream_model_AllPairedSynthetic_{l_idx}_{key}_{run}')
    os.remove(f'/data/imageGen/save/downstream_model_SubsetFirst_{l_idx}_{key}_{run}')
    os.remove(f'/data/imageGen/save/downstream_model_SubsetPairedReal_{l_idx}_{key}_{run}')
    os.remove(f'/data/imageGen/save/downstream_model_SubsetPairedSynthetic_{l_idx}_{key}_{run}')

    pickle.dump(stats, open(f'/data/imageGen/stats/trainingStats_{l_idx}_{key}_{run}.pkl', 'wb'))