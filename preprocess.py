import copy
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset
from torchtext.datasets import AmazonReviewFull
from torchtext.vocab import FastText


def download_data(path):
    AmazonReviewFull(root=path, split=('train', 'test'))


def load_data(path):
    df = pd.read_csv(path, header=None)  # , nrows=400)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df.rename({0: "star", 1: "rating1", 2: "rating2"}, axis=1, inplace=True)
    df["review"] = df["rating1"] + " " + df["rating2"]
    df.drop(columns=["rating1", "rating2"], inplace=True)
    df.star = df.star.apply(lambda x: int(x) - 1)
    return df


def train_test_split(df, test=0.3):
    idx = int(df.shape[0] * (1 - test))
    return df.iloc[:idx, :], df.iloc[idx:, :]


def preprocessing(sentence):
    """
    params sentence: a str containing the sentence we want to preprocess
    return the tokens list
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(str(sentence))
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return tokens


def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0


def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + (max_seq_len - len(list_of_indexes)) * [padding_index]
    return output[:max_seq_len]


def collate(batch, vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor([item[1] for item in batch])  # Use long tensor to avoid unwanted rounding
    return inputs, target


def get_processed_data(PIK, df, max_seq_len, save_freq=50, process=True):
    if process is True:
        # df = load_data(datafile)
        process_data(df, max_seq_len, PIK, save_freq)
    with open(PIK, "rb") as f:
        data = pickle.load(f)
    return Dataset(data)


def process_data(df, max_seq_len, PIK, save_freq):
    if not Path(PIK).is_file():
        vec = FastText()
        vec.vectors[1] = -torch.ones(vec.vectors[1].shape[0])
        vec.vectors[0] = torch.zeros(vec.vectors[0].shape[0])
        data = {'last_id': -1, 'row_shape': df.shape[0], 'sequences': [], 'max_seq_len': max_seq_len,
                'vec': vec, 'star': df.star}
        with open(PIK, "wb") as f:
            pickle.dump(data, f)

    with open(PIK, "rb") as f:
        data = pickle.load(f)

    if data['last_id'] < data['row_shape']:
        padded_seqs = []
        count = 0
        for sequence in df.review[data['last_id'] + 1:].tolist():
            seq = padding(encoder(preprocessing(sequence), data['vec']), max_seq_len)
            # print(len(seq),max_seq_len)
            padded_seqs.append(seq)
            count += 1
            if count % save_freq == 0:
                print('processing from id: (' + str(data['last_id'] + 1) + ':' + str(
                    data['last_id'] + save_freq) + ') of '
                      + str(data['row_shape']))
                data['sequences'].extend(padded_seqs)
                padded_seqs = []
                data['last_id'] += save_freq
                with open(PIK, "wb") as f:
                    pickle.dump(data, f)


class Dataset(Dataset):
    def __init__(self, data):  # df is the input df, max_seq_len is the max
        # lenght allowed to a sentence before cutting or padding
        self.max_seq_len = data['max_seq_len']
        self.vec = data['vec']
        self.vectorizer = self.get_vectorize
        self.labels = data['star']
        self.sequences = data['sequences']

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]

    def get_vectorize(self, x):
        return self.vec.vectors[x]


def train_loop(model, epochs, optimizer, criterion, train_loader, test_loader, emb_dim,
               printing_gap, saved_model_device, model_path, device, MAX_SEQ_LEN, PIK_plot_data):
    train_loss = []
    train_acc = []
    test_acc = []
    least_loss = np.inf

    for epoch in range(epochs):
        loss_train = 0
        # train_num_correct = 0
        # train_num_samples = 0

        for sentences, labels in train_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            sentences.resize_(sentences.size()[0], MAX_SEQ_LEN * emb_dim)

            optimizer.zero_grad()

            output = model.forward(sentences)  # 1) Forward pass
            loss = criterion(output, labels)  # 2) Compute loss
            loss.backward()  # 3) Backward pass
            optimizer.step()  # 4) Update model
            # _, predictions = output.max(1)
            #
            loss_train += loss.item()
            # train_num_correct += (predictions == labels).sum()
            # train_num_samples += predictions.size(0)

        model.eval()

        with torch.no_grad():
            train_num_correct = 0
            train_num_samples = 0
            for sentences, labels in iter(train_loader):
                sentences, labels = sentences.to(device), labels.to(device)
                sentences.resize_(sentences.size()[0], MAX_SEQ_LEN * emb_dim)

                output = model(sentences)
                _, predictions = output.max(1)
                train_num_correct += (predictions == labels).sum()
                train_num_samples += predictions.size(0)

        with torch.no_grad():
            test_num_correct = 0
            test_num_samples = 0
            for sentences, labels in iter(test_loader):
                sentences, labels = sentences.to(device), labels.to(device)
                sentences.resize_(sentences.size()[0], MAX_SEQ_LEN * emb_dim)

                output = model(sentences)
                _, predictions = output.max(1)

                test_num_correct += (predictions == labels).sum()
                test_num_samples += predictions.size(0)

        train_accu = float(train_num_correct) / train_num_samples * 100
        test_accu = float(test_num_correct) / test_num_samples * 100

        train_loss.append(loss_train / train_num_samples)
        train_acc.append(train_accu)
        test_acc.append(test_accu)

        # Save best model
        if loss_train < least_loss:
            least_loss = loss_train

            best_model_state = copy.deepcopy(model)
            best_model_state.to(saved_model_device)
            torch.save(best_model_state, model_path)

        if epoch % printing_gap == 0:
            print('Epoch: {}/{}\t.............'.format(epoch, epochs), end=' ')
            print("Train Loss: {:.4f}".format(loss_train / train_num_samples), end=' ')
            print("Train Acc: {:.4f}".format(train_accu), end=' ')
            print("Test Acc: {:.4f}".format(test_accu))

            # Save data to pickle
            data = {'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': test_acc}
            with open(PIK_plot_data, "wb") as f:
                pickle.dump(data, f)

        model.train()




if __name__ == '__main__':
    data_path = './data/AmazonReviewFull/amazon_review_full_csv/'

    df_train = load_data(data_path + "train.csv")
    df_test = load_data(data_path + "test.csv")
    batch_size = 16
    # print(df_test.shape)

    # # dataset_train = TrainData(df_train, max_seq_len=32)
    # dataset_test = TrainData(df_test, max_seq_len=32)
    #
    # # train_collate = lambda batch: collate(batch, vectorizer=dataset_train.vectorizer)
    # test_collate = lambda batch: collate(batch, vectorizer=dataset_test.vectorizer)
    #
    # # train_loader = DataLoader(dataset_train, batch_size=batch_size, collate_fn=train_collate)
    # test_loader = DataLoader(dataset_test, batch_size=batch_size, collate_fn=test_collate)
    # print(test_loader)
