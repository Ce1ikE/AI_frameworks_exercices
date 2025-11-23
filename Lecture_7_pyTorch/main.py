import torch
import torch.functional as F

BATCH_SIZE = 16
DIM_IN = 100       # typically higher values will be used (100)
HIDDEN_SIZE = 200  # typically higher values will be used (200)
DIM_OUT = 10

class MyModel(torch.nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        
        self.linear1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)
        self.softmax = torch.nn.Softmax()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # https://ai.stackexchange.com/questions/29982/why-do-we-add-1-in-the-formula-to-calculate-the-shape-of-the-output-of-the-convo
        # the values of the square convolution (aka the filter,aka the kernels) are the parameters the models tries to optimize during training 
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        # self.conv1 = torch.nn.Conv2d(1, 6, (5,5)) # similar notation, both width and height are identical
            # input : 32x32
            # window : 5x5
            # output : input - window + 1 = 32-5+1 = 28 for 2 dimensions and 6 output channels
        # A max_pool2D will then be applied: This will reduce the output to 14x14 (per channel, and there are 6 channels)
        # 6 input channels (received from the max_pool2D of 14x14), 16 output channels, 3x3 square convolution
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # self.conv2= torch.nn.Conv2d(6, 16, (3,3)) # similar notation, both width and height are identical
            # input : 14x14
            # window : 3x3
            # output 1 channel : input - window + 1 = 14-3+1 = 12 for 2 dimensions
            # output 16 channel : 16x12x12
        # A max_pool2D of (2x2) will take place: This will reduce the output to 6x6 (per channel, and there are 16 channels)
        # Fully connected layers  an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can also only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    pass

if __name__ == "__main__":
    main()
