train_loss = []
test_loss = []
train_loss1 = []
test_loss1 = []
train_loss2 = []
test_loss2 = []
test_iterloss = []
with torch.no_grad():
    model.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_iterloss.append(loss.item())



for epoch in range(1000):
    train_iterloss=[]
    test_iterloss = []
    train_iterloss1=[]
    test_iterloss1 = []
    train_iterloss2=[]
    test_iterloss2 = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(outputs, labels)

        train_iterloss.append(loss.item())
        train_iterloss1.append(loss1.item())
        train_iterloss2.append(loss2.item())

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
    with torch.no_grad():
        model.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(outputs, labels)

        test_iterloss.append(loss.item())
        test_iterloss1.append(loss1.item())
        test_iterloss2.append(loss2.item())

    train_loss.append(sum(train_iterloss)/len(train_iterloss))
    test_loss.append(sum(test_iterloss)/len(test_iterloss))
    train_loss1.append(sum(train_iterloss1)/len(train_iterloss1))
    test_loss1.append(sum(test_iterloss1)/len(test_iterloss1))
    train_loss2.append(sum(train_iterloss2)/len(train_iterloss2))
    test_loss2.append(sum(test_iterloss2)/len(test_iterloss2))


