optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(2):
    iterloss=[]
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        iterloss.append(loss.item())

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    print('epoch=',epoch,'train loss=', sum(iterloss)/len(iterloss))

    with torch.no_grad():
        model.eval()

        

a, b = test_dataset[0:95]
model.eval()
out = model(a.to(device))
out = scaler_y.inverse_transform(out.cpu().detach().numpy())
b = scaler_y.inverse_transform(b)

modelsc = LinearRegression()
modelsc.fit(a, b)
r_squared = modelsc.score(a, b)
print(f'R-Squared is: {r_squared}')
