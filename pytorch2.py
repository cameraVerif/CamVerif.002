import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = (self.relu(self.conv1(x)))
        x = (self.relu(self.conv2(x)))
        x = (self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

model = SimpleCNN()
model.load_state_dict(torch.load('cifar_cnn.pth'))
model.eval()



import torch.onnx

# Create a dummy input tensor with the same shape as the model's input
dummy_input = torch.randn(1, 3, 32, 32)

# Export the model
torch.onnx.export(
    model,                   # Model to be exported
    dummy_input,             # Dummy input tensor
    "cifar_cnn.onnx",        # Name of the output ONNX file
    export_params=True,      # Store the trained parameter weights inside the model file
    opset_version=11,        # ONNX version to export the model to
    do_constant_folding=True # Whether to execute constant folding for optimization
)

print("Model has been converted to ONNX format")
