class CustomDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = torch.tensor(data)
        self.target = torch.tensor(target)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        array = self.data[idx,:]
        label = self.target[idx]
        if self.transform:
            array = self.transform(array)
        if self.target_transform:
            label = self.target_transform(label)
        return array, label

train_loader = torch.utils.data.DataLoader(train_dataset , shuffle=True, batch_size=256)
test_loader  = torch.utils.data.DataLoader(test_dataset, shuffle=True,  batch_size=16)

class BN_net(nn.Module):
    def __init__(self):
        super(BN_net, self).__init__()
        self.conv1 = nn.Conv1d(my_len, 20, 3) #input = (batchsize, my_len=5, 11) output = (
        self.conv2 = nn.Conv1d(20, 40, 3)
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(240, 100)
        self.fc2 = nn.Linear(100, 1)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(40)

    def forward(self, x):
        x = self.conv1(x.view([-1, 1, 10]))
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout1(x)
        #print(x.shape)
        #x = torch.flatten(x, 0)
        x = self.fc1(x.view(-1,240))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class my_net(nn.Module):
    def __init__(self):
        super(my_net, self).__init__()
        self.conv1 = nn.Conv1d(my_len, 20, 3) #input = (batchsize, my_len=5, 11) output = (
        self.conv2 = nn.Conv1d(20, 40, 3)
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(120, 100)
        self.fc2 = nn.Linear(100, 1)


    def forward(self, x):
        x = self.conv1(x.view([-1, 1, 7]))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #print(x.shape)
        #x = torch.flatten(x, 0)
        x = self.fc1(x.view(-1,120))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class my_net1(nn.Module):
    def __init__(self):
        super(my_net1, self).__init__()

        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x.view([-1, 10]))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #print(x.shape)
        #x = torch.flatten(x, 0)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc5(x)
        return x

class my_net2(nn.Module):
    def __init__(self):
        super(my_net2, self).__init__()
        self.conv1 = nn.Conv1d(my_len, 20, 3) #input = (batchsize, my_len=5, 11) output = (
        self.conv2 = nn.Conv1d(20, 100, 3)
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(600, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x.view([-1, 1, 10]))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #print(x.shape)
        #x = torch.flatten(x, 0)
        x = self.fc1(x.view(-1,600))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class my_net3(nn.Module):
    def __init__(self):
        super(my_net3, self).__init__()
        self.conv1 = nn.Conv1d(my_len, 10, 3) #input = (batchsize, my_len=5, 11) output = (
        self.conv2 = nn.Conv1d(10, 20, 3)
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(120, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x.view([-1, 1, 10]))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #print(x.shape)
        #x = torch.flatten(x, 0)
        x = self.fc1(x.view(-1,120))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class fc_net(nn.Module):
    def __init__(self):
        super(fc_net, self).__init__()

        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1000)
        self.fc3 = nn.Linear(1000, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

model = my_net().double()
model = model.to(device)

