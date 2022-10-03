import torch
import torch.optim as optim
import random
import torch.nn as nn
from util import load_unsupervised_data_n_model
import argparse
from torch.autograd import Variable


class EntLoss(nn.Module):
    def __init__(self, args, lam1, lam2, pqueue=None):
        super(EntLoss, self).__init__()
        self.lam1 = lam1
        self.lam2 = lam2
        self.pqueue = pqueue
        self.args = args
    
    def forward(self, feat1, feat2, use_queue=False):
        probs1 = torch.nn.functional.softmax(feat1, dim=-1)
        probs2 = torch.nn.functional.softmax(feat2, dim=-1)
        loss = dict()
        loss['kl'] = 0.5 * (KL(probs1, probs2, self.args) + KL(probs2, probs1, self.args))

        sharpened_probs1 = torch.nn.functional.softmax(feat1/self.args.tau, dim=-1)
        sharpened_probs2 = torch.nn.functional.softmax(feat2/self.args.tau, dim=-1)
        loss['eh'] = 0.5 * (EH(sharpened_probs1, self.args) + EH(sharpened_probs2, self.args))

        # whether use historical data
        loss['he'] = 0.5 * (HE(sharpened_probs1, self.args) + HE(sharpened_probs2, self.args))
        
        # TWIST Loss
        loss['final'] = loss['kl'] + ((1+self.lam1)*loss['eh'] - self.lam2*loss['he'])
        
        #########################################################################
        # probability distribution (PKT by Kernel Density Estimation)
        loss['kde'] = cosine_similarity_loss(feat1, feat2)
        
        # nuclear-norm
        loss['n-norm'] = -0.5 * (torch.norm(sharpened_probs1,'nuc')+torch.norm(sharpened_probs2,'nuc')) * 0.001        
        
        loss['final-kde'] = loss['kde'] * 100 + loss['final']#+ loss['n-norm']

        return loss

def KL(probs1, probs2, args):
    kl = (probs1 * (probs1 + args.EPS).log() - probs1 * (probs2 + args.EPS).log()).sum(dim=1)
    kl = kl.mean()
    return kl

def CE(probs1, probs2, args):
    ce = - (probs1 * (probs2 + args.EPS).log()).sum(dim=1)
    ce = ce.mean()
    return ce

def HE(probs, args): 
    mean = probs.mean(dim=0)
    ent  = - (mean * (mean + args.EPS).log()).sum()
    return ent

def EH(probs, args):
    ent = - (probs * (probs + args.EPS).log()).sum(dim=1)
    mean = ent.mean()
    return mean

def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss

def gaussian_noise(csi, epsilon):
    noise = torch.normal(1, 2, size=(3, 114, 500)).cuda()
    perturbed_csi = csi + epsilon*noise
    return perturbed_csi


def main():
    learning_rate = 1e-3
    parser = argparse.ArgumentParser('Self-Supervised')
    parser.add_argument('--tau', type=float, default=1.0, metavar='LR')
    parser.add_argument('--EPS', type=float, default=1e-5, help='episillon')
    parser.add_argument('--weight-decay', type=float, default=1.5e-6, help='weight decay (default: 1e-4)')
    parser.add_argument('--lam1', type=float, default=0.0, metavar='LR')
    parser.add_argument('--lam2', type=float, default=1.0, metavar='LR')
    parser.add_argument('--local_crops_number', type=int, default=12)
    parser.add_argument('--min1', type=float, default=0.4, metavar='LR')
    parser.add_argument('--max1', type=float, default=1.0, metavar='LR')
    parser.add_argument('--min2', type=float, default=0.05, metavar='LR')
    parser.add_argument('--max2', type=float, default=0.4, metavar='LR')
    parser.add_argument('--gpu', type=int, default=1, metavar='gpu')
    parser.add_argument('--eval', type=str, default='no', metavar='gpu')
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM','CNN+GRU','ViT'])
    args = parser.parse_args()
    args.global_crops_scale = (args.min1, args.max1)
    args.local_crops_scale = (args.min2, args.max2)

    criterion = EntLoss(args, 0.0, 0.5)

    root = "./Data/"
    unsupervised_train_loader, supervised_train_loader, test_dataloader, model = load_unsupervised_data_n_model(args.model,root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model.to(device)
    #######################################
    # self-supervised training
    print ('Self-supervised encoder training')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    for epoch in range(100):
            total_loss = 0
            kl_loss = 0
            eh_loss = 0
            he_loss = 0
            kde_loss = 0
            for data in unsupervised_train_loader:
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    x1 = gaussian_noise(x, random.uniform(0, 2.0))
                    x2 = gaussian_noise(x, random.uniform(0.1, 2.0))

                    # ===================forward=====================
                    feat_x1, feat_x2 = model(x1, x2)
                    loss = criterion(feat_x1, feat_x2)
                    loss_kl = loss['kl']
                    loss_eh = loss['eh']
                    loss_he = loss['he']
                    loss_kde = loss['kde']
                    loss = loss['final-kde']

                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # ===================log========================
                    total_loss += loss.data
                    kl_loss += loss_kl.data
                    eh_loss += loss_eh.data
                    he_loss += loss_he.data
                    kde_loss += loss_kde.data
            print('epoch [{}/{}], total loss:{:.4f},kl loss:{:.4f},eh loss:{:.4f},he loss:{:.4f},kde loss:{:.4f}'
                    .format(epoch+1,100, total_loss, kl_loss, eh_loss, he_loss, kde_loss))
    #######################################

    #######################################
    # test
    def test():
        model.eval()
        correct_1, correct_2 = 0, 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                x, y = data
                x, y = x.to(device), y.to(device)

                y1, y2 = model(x, x, flag='supervised')
                _, pred_1 = torch.max(y1.data, 1)
                _, pred_2 = torch.max(y2.data, 1)
                total += y.size(0)
                correct_1 += (pred_1 == y).sum().item()
                correct_2 += (pred_2 == y).sum().item()

        print('Test accuracy: {:.2f}%, {:.2f}%'.format(100 * correct_1 / total, 100 * correct_2 / total))
    #######################################

    ##################################
    # supervised learning
    print ('Supervised classifier training')
    optimizer_supervised = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-5)
    ce_criterion = nn.CrossEntropyLoss()

    for epoch in range(300):
        model.train()
        total_loss = 0
        for data in supervised_train_loader:
            x, y = data
            x = Variable(x).to(device)
            y = y.type(torch.LongTensor)
            y = y.to(device)

            # ===================forward=====================
            y1, y2 = model(x, x, flag='supervised')
            loss = ce_criterion(y1, y) + ce_criterion(y2, y)

            # ===================backward====================
            optimizer_supervised.zero_grad()
            loss.backward()
            optimizer_supervised.step()
        # ===================log========================
        total_loss += loss.data
        print('epoch [{}/{}], loss:{:.6f}'
            .format(epoch+1, 300, total_loss))
        # test
        if epoch > 250:
            test()
    ##################################
    return

if __name__ == "__main__":
    main()
