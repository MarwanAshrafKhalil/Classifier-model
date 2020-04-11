epochs =3
steps = 0
running_loss = 0
print_every = 40
print("hello")

for e in range(epochs):
    model.train()
    model = model.to('cuda')
    for images, labels in train_dataloaders:
        
        steps += 1
        images, labels = images.to(device), labels.to(device)
#         print(images.size())
        # Flatten images into a 784 long vector
#         print(images.size())
#         print(images.size())
        
#         print(images.size())
        optimizer.zero_grad()
        
        output = model.forward(images)        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, valid_dataloaders, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(train_dataloaders)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(train_dataloaders)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()
            
            
            
            ___________________________________
            
            def validation(model, valid_dataloaders, criterion):
    test_loss = 0
    accuracy = 0
    model = model.to('cuda')
    for images, labels in valid_dataloaders:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy
