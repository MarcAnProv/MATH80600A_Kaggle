import time
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import (
    create_embedding_layer,
    create_vocabulary,
    embedding_matrix,
    get_logger,
    make_sets
)


logger = get_logger()


class Dataset(Dataset):
    """Custom dataset concatenating abstract and title in one feature"""
    def __init__(self, data):
        self.x_title = torch.from_numpy(data.iloc[:, 1])
        self.x_abstract = torch.from_numpy(data.iloc[:, 2])
        self.y = torch.from_numpy(data.iloc[:, -1])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.cat((self.x_title, self.x_abstract), dim=1)[idx], self.y[idx]


class GRUClassifier(nn.Module):
    # TODO: freeze embeddings for first epoch, then unfreeze
    def __init__(self, embedding_matrix, hidden_dim, label_size, num_layers, device):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.embedding, embedding_dim = create_embedding_layer(embedding_matrix)
        self.num_layers = num_layers
        self.device = device
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, label_size)

    def zero_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)

    def forward(self, features):
        batch_size = features.size(0)
        embedding = self.embedding(features)
        h_0 = self.zero_state(batch_size)
        output, hidden = self.gru(embedding, h_0)
        logger.info(f"output size {output.size()}")
        output = self.fc(output[:, -1, :])
        return output


def _train_one_epoch(model, train_loader, optimizer, criterion, epoch, device):
    total_loss, total_correct, total_prediction = 0.0, 0.0, 0.0
    for x, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(x)
        predictions = torch.max(logits, dim=-1)[1]
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += torch.eq(predictions, label).sum().item()
        total_prediction += label.size(0)
    return total_loss / len(train_loader), total_correct / total_prediction


def _evaluate_one_epoch(model, valid_loader, criterion, device):
    total_loss, total_correct, total_prediction = 0.0, 0.0, 0.0
    model.eval()
    criterion.to(device)
    with torch.no_grad():
        for x, label in valid_loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            predictions = torch.max(logits, dim=-1)[1]
            loss = criterion(logits, label)

            total_loss += loss.item()
            total_correct += torch.eq(predictions, label).sum().item()
            total_prediction += label.size(0)
    return total_loss / len(valid_loader), total_correct / total_prediction


def train(model, train_loader, valid_loader, epoch, device):
    logger.info("Training loop started")
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)
    model.train()
    time_start = time.time()
    for epochs in range(epoch):
        train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, criterion, epoch, device)
        val_loss, val_acc = _evaluate_one_epoch(model, valid_loader, criterion, device)
        logger.info('Epoch {} | Train loss {:.3f} | Valid loss {:.3f} | Valid acc {:.3f}'.format(epoch, train_loss, val_loss, val_acc))
    time_elapsed = time.time() - time_start()
    logger.info(f"Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.2f}s\n")


def main():
    training_set, local_test_set, _ = make_sets("/data/train.csv", "/data/text.csv", "/data/nodeid2paperid.csv", "/data/test.csv", 0.1)
    vocabulary = create_vocabulary(training_set)
    embed_matrix = embedding_matrix(vocabulary)
    training_set = Dataset(training_set)
    local_test_set = Dataset(local_test_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(training_set, batch_size=32)
    valid_loader = DataLoader(local_test_set, batch_size=32)
    model = GRUClassifier(embed_matrix, 128, 20, 2, device)
    train(model, train_loader, valid_loader, 25, device)


if __name__ == "__main__":
    main()
