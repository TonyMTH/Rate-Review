import torch
from torch import nn, optim

# Don't change this once you started preprocessing Data to avoid dim mismatch
# MAX_SEQ_LEN = 32
emb_dim = 300
max_seq_len = 32

lr = 0.003
criterion = nn.CrossEntropyLoss()
Optimizer = lambda x: optim.SGD(x, lr=lr, momentum=0.6)

hidden1, hidden2, hidden3 = 256, 256, 256

epochs = 500
printing_gap = 1
model_path = 'data/best_model.pt'
saved_model_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
PIK = "data/pickle.dat"
PIK_train = "data/train_pickel.dat"
PIK_test = "data/test_pickel.dat"

data_path = './data/AmazonReviewFull/amazon_review_full_csv/'
data_train = data_path + "train.csv"
data_test = data_path + "test.csv"

PIK_plot_data = './data/plot_data.dat'
