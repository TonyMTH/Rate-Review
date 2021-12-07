from torch import nn
import torch.nn.functional as F
import parameters as p


class Classifier(nn.Module):
    def __init__(self, max_seq_len, emb_dim, output_dim, hidden1=16, hidden2=16, hidden3=16):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(max_seq_len * emb_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_dim)
        self.out = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.squeeze(1).float()))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        return self.out(x)


Classifier2 = lambda x: nn.Sequential(nn.Linear(p.max_seq_len * p.emb_dim, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Linear(256, 500),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Linear(500, 500),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Linear(500, x),
                                      nn.LogSoftmax(dim=1)
                                      )

Classifier3 = lambda x: nn.Sequential(
    nn.Linear(p.max_seq_len * p.emb_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, x),
    nn.LogSoftmax(dim=1)
)
