import argparse
import os

import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='path to the individual data files in .npz format')
    args = parser.parse_args()
    return args


def create_seq_data(features, targets, n=5):
    features = np.column_stack([features[i: len(features) - n + i, :] for i in range(n)])
    targets = targets[n:]

    return features, targets


def normalize(X, means=None, stds=None):
    if means is None:
        means = np.mean(X, axis=0)
    if stds is None:
        stds = np.std(X, axis=0)

    return (X - means) / stds, means, stds


def load_data(root_path, cond, n=5):
    participant_dir = [f for f in os.listdir(root_path) if cond(f)]

    feature_list = []
    target_list = []

    for participant in participant_dir:
        npz_filenames = os.listdir(os.path.join(root_path, participant))
        for npz_filename in npz_filenames:
            data = np.load(os.path.join(root_path, participant, npz_filename))

            features = data['features'][:, 1:]
            targets = data['targets']

            features, targets = create_seq_data(features, targets, n=n)

            feature_list.append(features)
            target_list.append(targets)


    features = np.row_stack(feature_list)
    targets = np.row_stack(target_list)

    targets = np.argmax(targets, axis=-1)

    return features, targets


class SimpleModel(torch.nn.Module):
    def __init__(self, n):
        super(SimpleModel, self).__init__()
        self.n = n
        self.linear_1 = torch.nn.Linear(n * 18, 256)
        self.linear_2 = torch.nn.Linear(256, 256)
        self.linear_3 = torch.nn.Linear(256, 256)
        self.linear_4 = torch.nn.Linear(256, 6)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = torch.relu(x)
        x = self.linear_3(x)
        x = torch.relu(x)
        x = self.linear_4(x)

        return x


def train_and_evaluate(root_path):
    # cond_val = lambda x: '2' in x
    # cond_train = lambda x: '0' in x or '1' in x
    cond_val = lambda x: 'jano' in x or 'zuzka' in x or 'iveta' in x # or 'stefan' in x or 'palo' in x
    # cond_val = lambda x: 'viktor' in x #or 'zuzka' in x or 'iveta' in x or 'stefan' in x or 'palo' in x
    cond_train = lambda x: not cond_val(x)

    # for person in ['janci', 'viktor', 'igor', 'barbora', 'zdenka']:
    #     cond_train = lambda x: ('0' in x or '1' in x) and person not in x
    #     cond_val = lambda x: ('0' in x or '1' in x) and person in x

        # cond_train = lambda x: 'viktor' not in x
        # cond_val = lambda x: 'viktor' in x

    n = 10

    train_X, train_y = load_data(root_path, cond_train, n=n)
    train_X, means, stds = normalize(train_X)

    val_X, val_y = load_data(root_path, cond_val, n=n)

    val_X, _, _ = normalize(val_X, means, stds)

    print(f"Loaded data with {len(train_X)} training samples and {len(val_X)} test samples")

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_X.astype(np.float32)), torch.tensor(train_y, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_X.astype(np.float32)), torch.tensor(val_y, dtype=torch.long))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)

    model = SimpleModel(n).cuda()
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    train_loss_running = 0.0

    for e in range(30):
        for X, y in train_loader:
            pred = model(X.cuda())
            y = y.cuda()
            optimizer.zero_grad()

            loss = ce_loss(pred, y)
            loss.backward()
            optimizer.step()

            train_loss_running = 0.9 * train_loss_running + 0.1 * loss
            # print("Running loss: {}".format(train_loss_running.item()))

        with torch.no_grad():
            y_pred = []
            y_true = []


            for X, y in val_loader:
                pred = model(X.cuda())
                y = y.cuda()

                # if e > 5:
                #     print(pred)

                y_pred.extend(torch.argmax(pred, dim=-1).detach().cpu().numpy())
                y_true.extend(y.detach().cpu().numpy())

            print("Validation f1: {}".format(f1_score(y_true, y_pred, average='weighted')))

    print(classification_report(y_true, y_pred))



if __name__ == '__main__':
    args = parse_args()
    train_and_evaluate(args.root_path)