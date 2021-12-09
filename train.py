from torch import nn, optim
from torch.utils.data import DataLoader
import model as md
import preprocess as pr
from parameters import *

# Define Processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1.\t"+str(device.type).capitalize()+" detected")

# Fetch Data
df_train = pr.load_data(data_train)
df_test = pr.load_data(data_test)
output_dim = len(df_train.star.unique())

# Define Model
model = md.Model1(max_seq_len, emb_dim, hidden1, hidden2, hidden3, output_dim)
# model = md.Model2(output_dim)
model.to(device)
print("2.\tModel defined and moved to " + str(device.__str__()))

# Parameters
optimizer = Optimizer(model.parameters())
print("3.\tCriterion set as " + str(criterion.__str__()))
print("4.\tOptimizer set as " + str(optimizer.__str__()))

# Loaders
# dataset_train, dataset_test = pr.get_load_train_ata(PIK, data_train, data_test, max_seq_len)
print('5.\tprocessing train data.......')
dataset_train = pr.get_processed_data(PIK_train, df_train, max_seq_len, save_freq=200, process=False)
print('6.\tprocessing test data.......')
dataset_test = pr.get_processed_data(PIK_test, df_test, max_seq_len, save_freq=200, process=False)

print("7.\tCollate")
train_collate = lambda batch: pr.collate(batch, vectorizer=dataset_train.vectorizer)
test_collate = lambda batch: pr.collate(batch, vectorizer=dataset_test.vectorizer)

print("8.\tDataLoader")
train_loader = DataLoader(dataset_train, batch_size=batch_size, collate_fn=train_collate)
test_loader = DataLoader(dataset_test, batch_size=batch_size, collate_fn=test_collate)

# Train Model
print("9.\tTrain loop")
pr.train_loop(model, epochs, optimizer, criterion, train_loader, test_loader, emb_dim,
              printing_gap, saved_model_device, model_path, device, max_seq_len, PIK_plot_data)
