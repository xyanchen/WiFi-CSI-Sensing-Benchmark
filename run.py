import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model
import os
from torch.utils.tensorboard import SummaryWriter

from confusionMatrix import ConfusionMatrix

import msvcrt
from torchsummary import summary
from thop import profile

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device, folder):
    writer = SummaryWriter(folder + '/runs')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
    
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        if msvcrt.kbhit():  # 检查是否有键盘输入
            user_input = msvcrt.getch().decode('utf-8').strip()
            if user_input == 'q':
                print("Stopping training after this epoch.")
                break

    writer.close()
    torch.save(model.state_dict(), folder + '/model.pth')
    return

def test(model, tensor_loader, criterion, device, folder, num_classes, labels):
    print('Testing')
    model_path = folder + '/model.pth'
    assert os.path.exists(model_path), "cannot find {} file".format(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)
    
    model.eval()
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, test_labels = data
        inputs = inputs.to(device)
        test_labels.to(device)
        test_labels = test_labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,test_labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == test_labels.to(device)).sum().item() / test_labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)

        confusion.update(predict_y.to("cpu").numpy(), test_labels.to("cpu").numpy())
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))

    confusion.summary()
    confusion.plot(folder)
    return
    
def main():
    root = r'D:/WiFi-CSI-Sensing-Benchmark/Data/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT','oneD_lw_ResNet18','oneD_lw_ResNet50'])
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch, num_classes, labels = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    folder = './project/' + args.dataset + '_' + args.model
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('The folder is {}'.format(folder))

    # summary(model,input_size=(3,114,500))
    # inputs = torch.randn(1, 3, 114,500).to(device)
    # flops, params = profile(model, (inputs,))
    # print('flops: ', flops, 'params: ', params)
    # exit()

    train(
        model=model,
        tensor_loader= train_loader,
        num_epochs= train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
        folder=folder
         )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device,
        folder=folder,
        num_classes=num_classes,
        labels=labels
        )
    return


if __name__ == "__main__":
    main()
