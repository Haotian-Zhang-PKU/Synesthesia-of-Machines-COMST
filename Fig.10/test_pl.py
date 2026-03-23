import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as sciio
from data_feed import MatSequenceDataset

# ======================
# Model Definition
# ======================
DROPOUT_A = 0
DROPOUT_B = 0
N = 100 #change

class Linear_net(nn.Module):
    def __init__(self):
        super(Linear_net, self).__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(42, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_A),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_B)
        )
        self.output_layer = nn.Linear(128, 52)
    def forward(self, x):
        x = self.layer(x)
        x = x.reshape(x.size(0), -1)
        output = self.output_layer(x)
        return output


# ======================
# Config
# ======================
TEST_DATA_PATH = r'.\test'
LABEL_SAVE_DIR = r'.\label_test_test'
OUTPUT_SAVE_DIR = r'.\output_test_test'

BATCH_SIZE = 1
MODEL_PATH = 'model.params'


# ======================
# Data Loader
# ======================
test_dataset = MatSequenceDataset(TEST_DATA_PATH, use_natural_sort=True)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)


# ======================
# Model Load
# ======================
net = Linear_net()
net.load_state_dict(torch.load(MODEL_PATH))
net.eval()

loss_fn = nn.MSELoss()

print('==== Start Evaluation ====')

total_mse = 0.0
num_samples = 0


# ======================
# Evaluation Loop
# ======================
with torch.no_grad():
    for batch_idx, (features, targets) in enumerate(test_dataloader):

        file_id = batch_idx + 1
        num_samples += 1

        # reshape & type cast
        features = features.view(BATCH_SIZE, 42).float()
        targets = torch.squeeze(targets, 1)

        # forward
        outputs = net(features)

        # save .mat files
        label_file = rf'{LABEL_SAVE_DIR}\label_test{file_id}.mat'
        output_file = rf'{OUTPUT_SAVE_DIR}\output_test{file_id}.mat'

        sciio.savemat(label_file, {'label_test': targets.tolist()})
        sciio.savemat(output_file, {'output_test': outputs.tolist()})

        # logging
        print(f"[Sample {file_id}] Prediction:\n{outputs}")

        # loss
        mse = loss_fn(outputs, targets)
        total_mse += mse

        print(f"[Sample {file_id}] MSE: {mse}\n")


# ======================
# Final Result
# ======================
average_mse = total_mse / N

print("==== Evaluation Finished ====")
print(f"Average MSE Loss: {average_mse:.4e}")
print(f"Total Samples Processed: {num_samples}")
