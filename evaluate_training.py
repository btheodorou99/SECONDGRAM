import torch
import pickle
import numpy as np 
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics

# Train models on:
# 1. All first images
# 2. All paired images (real where present, imputed where not)
# 3. All paired images (imputed only)
# 4. Subset (where there is a second image) first images
# 5. Subset paired images (real second)
# 6. Subset paired images (imputed second)

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
generatedValData = pickle.load(open(f'/data/theodoroubp/imageGen/generatedValData_{key}.pkl', 'rb'))
testData = pickle.load(open('/data/theodoroubp/imageGen/testData.pkl', 'rb'))
LABEL_IDX = 3
EPOCHS = 250
PATIENCE = 5
LR = 1e-4
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train1 = [(d[1], d[0][LABEL_IDX]) for d in generatedTrainData]
train2 = [(torch.cat((d[1], d[2] if d[2] is not None else d[3])), d[0][LABEL_IDX]) for d in generatedTrainData]
train3 = [(torch.cat((d[1], d[3])), d[0][LABEL_IDX]) for d in generatedTrainData]
train4 = [(d[1], d[0][LABEL_IDX]) for d in generatedTrainData if d[2] is not None]
train5 = [(torch.cat((d[1], d[2])), d[0][LABEL_IDX]) for d in generatedTrainData if d[2] is not None]
train6 = [(torch.cat((d[1], d[3])), d[0][LABEL_IDX]) for d in generatedTrainData if d[2] is not None]
val1 = [(d[1], d[0][LABEL_IDX]) for d in generatedValData]
val2 = [(torch.cat((d[1], d[2] if d[2] is not None else d[3])), d[0][LABEL_IDX]) for d in generatedValData]
val3 = [(torch.cat((d[1], d[3])), d[0][LABEL_IDX]) for d in generatedValData]
val4 = [(d[1], d[0][LABEL_IDX]) for d in generatedValData if d[2] is not None]
val5 = [(torch.cat((d[1], d[2])), d[0][LABEL_IDX]) for d in generatedValData if d[2] is not None]
val6 = [(torch.cat((d[1], d[3])), d[0][LABEL_IDX]) for d in generatedValData if d[2] is not None]
testDataSingle = [(d[1], d[0][LABEL_IDX]) for d in testData if d[2] is not None]
testDataPaired = [(torch.cat((d[1], d[2])), d[0][LABEL_IDX]) for d in testData if d[2] is not None]

class DownstreamModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=256, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, output_dim)

    def forward(self, input):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(input)))))


def get_batch(dataset, loc, batch_size):
    data = dataset[loc:loc+batch_size]
    bs = len(data)
    batch_input = torch.zeros(bs, len(data[0][0]))
    batch_labels = torch.zeros(bs, 1)
    for i, d in enumerate(data):
        batch_input[i] = d[0]
        batch_labels[i] = d[1]

    return batch_input.to(DEVICE), batch_labels.to(DEVICE)

def train_model(model, label, trainDataset, valDataset):
    print(label)
    patience = 0
    global_loss = 1e10
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for e in tqdm(range(EPOCHS), leave=False):
        np.random.shuffle(trainDataset)
        train_losses = []
        model.train()
        for i in tqdm(range(0, len(trainDataset), BATCH_SIZE), leave=False):
            batch_input, batch_labels = get_batch(trainDataset, i, BATCH_SIZE)
            batch_preds = model(batch_input)
            loss = mse(batch_preds, batch_labels)
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
                loss = mse(batch_preds, batch_labels)
                val_losses.append(loss.detach().cpu().item())

            currTrainLoss = np.mean(train_losses)
            currValLoss = np.mean(val_losses)
            # print(f'\tEpoch: {e}; Train Loss: {currTrainLoss}; Validation Loss: {currValLoss}')
            if currValLoss < global_loss:
                patience = 0
                global_loss = currValLoss
                torch.save(model.state_dict(), f'/data/theodoroubp/imageGen/save/downstream_model_{label}')
            else:
                patience += 1
                if patience == PATIENCE:
                    break

    model.load_state_dict(torch.load(f'/data/theodoroubp/imageGen/save/downstream_model_{label}', map_location='cpu'))
    return model

def test_model(model, testDataset):
    preds = []
    labels = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(testDataset), BATCH_SIZE), leave=False):
            batch_input, batch_labels = get_batch(testDataset, i, BATCH_SIZE)
            batch_preds = model(batch_input).relu()
            preds += batch_preds.squeeze(-1).cpu().tolist()
            labels += batch_labels.squeeze(-1).cpu().tolist()

    results = {
        'MAE': metrics.mean_absolute_error(labels, preds),
        'MSE': metrics.mean_squared_error(labels, preds)
    }

    print(results)
    return results

results = {}

model = DownstreamModel(len(train1[0][0])).to(DEVICE)
model = train_model(model, 'AllFirst', train1, val1)
results['AllFirst'] = test_model(model, testDataSingle)

model = DownstreamModel(len(train2[0][0])).to(DEVICE)
model = train_model(model, 'AllPairedImputed', train2, val2)
results['AllPairedImputed'] = test_model(model, testDataPaired)

model = DownstreamModel(len(train3[0][0])).to(DEVICE)
model = train_model(model, 'AllPairedSynthetic', train3, val3)
results['AllPairedSynthetic'] = test_model(model, testDataPaired)

model = DownstreamModel(len(train4[0][0])).to(DEVICE)
model = train_model(model, 'SubsetFirst', train4, val4)
results['SubsetFirst'] = test_model(model, testDataSingle)

model = DownstreamModel(len(train5[0][0])).to(DEVICE)
model = train_model(model, 'SubsetPairedReal', train5, val5)
results['SubsetPairedReal'] = test_model(model, testDataPaired)

model = DownstreamModel(len(train6[0][0])).to(DEVICE)
model = train_model(model, 'SubsetPairedSynthetic', train6, val6)
results['SubsetPairedSynthetic'] = test_model(model, testDataPaired)

pickle.dump(results, open(f'/data/theodoroubp/imageGen/results_{key}.pkl', 'wb'))
