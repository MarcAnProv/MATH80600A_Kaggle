import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from src.utils import (
    get_logger,
    make_predictions_csv,
    process_text
)


logger = get_logger()


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_size, num_layers, device):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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
        output = self.fc(output[:, -1, :])
        return output


def _train_one_epoch(model, iterator, optimizer, criterion, epoch, device):
    total_loss, total_correct, total_prediction = 0.0, 0.0, 0.0
    model.train()
    for batch in tqdm(iterator, total=len(iterator), desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        logits = model(batch.text)
        predictions = torch.max(logits, dim=-1)[1]
        loss = criterion(logits, batch.label.long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += torch.eq(predictions, batch.label.long()).sum().item()
        total_prediction += batch.label.size(0)
    return total_loss / len(iterator), total_correct / total_prediction


def _evaluate_one_epoch(model, iterator, criterion, device):
    total_loss, total_correct, total_prediction = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            logits = model(batch.text)
            predictions = torch.max(logits, dim=-1)[1]
            loss = criterion(logits, batch.label.long())

            total_loss += loss.item()
            total_correct += torch.eq(predictions, batch.label.long()).sum().item()
            total_prediction += batch.label.size(0)
    return total_loss / len(iterator), total_correct / total_prediction


def train(model, train_iterator, valid_iterator, epoch, device):
    logger.info("Training loop started")
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # decay lr by 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.to(device)
    criterion.to(device)
    model.train()
    time_start = time.time()
    for epochs in range(epoch):
        train_loss, train_acc = _train_one_epoch(model, train_iterator, optimizer, criterion, epoch, device)
        scheduler.step()
        logger.info(f"scheduler lr: {scheduler.get_last_lr()}")
        val_loss, val_acc = _evaluate_one_epoch(model, valid_iterator, criterion, device)
        logger.info(
            'Epoch {} | Train loss {:.3f} | Valid loss {:.3f} | Valid acc {:.3f}'.format(epoch, train_loss, val_loss, val_acc)
        )
    time_elapsed = time.time() - time_start
    logger.info(f"Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.2f}s\n")


def predict(model, test_iterator, device):
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in test_iterator:
            logits = model(batch.text)
            predictions = torch.max(logits, dim=-1)[1]
            preds.append(predictions)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_dim",
        help="Dimension of the embeddings",
        type=int,
        default=100
    )
    parser.add_argument(
        "--hidden_dim",
        help="Number of hidden nodes per layer",
        type=int
    )
    parser.add_argument(
        "--output_dim",
        help="Output dimension",
        type=int,
        default=20
    )
    parser.add_argument(
        "--num_layers",
        help="Number of GRU layers",
        type=int
    )
    parser.add_argument(
        "--epochs",
        help="Number of training epochs",
        type=int,
        default=25
    )
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator, vocab_size, pretrained_embeddings = process_text(device)
    model = GRUClassifier(vocab_size, args.embedding_dim, args.hidden_dim, args.output_dim, args.num_layers, device)
    # load pre-trained embeddings and freeze them
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.requires_grad = False
    train(model, train_iterator, valid_iterator, args.epochs, device)
    logger.info("Making predictions and writing them to csv")
    preds = predict(model, test_iterator, device)
    make_predictions_csv("/data/transformed_test.csv", preds)


if __name__ == "__main__":
    main()
