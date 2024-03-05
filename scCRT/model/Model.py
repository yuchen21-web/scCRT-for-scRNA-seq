import numpy as np
import torch
from torch import nn, optim

class scCRT(nn.Module):
    def __init__(self, input_size, middle_size, hidden_size):
        super(scCRT, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, hidden_size),
            #             nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, input_size),
            #             nn.ReLU()
        )

    def forward(self, x):
        output = self.encoder(x)
        recon = self.decoder(output)
        return output, recon

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class Cell_pair_loss(nn.Module):
    def __init__(self):
        super(Cell_pair_loss, self).__init__()

        self.BCE_loss = nn.BCELoss(reduction='mean')

        self.euclid = nn.PairwiseDistance(p=2)

        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, y, pos_cell, neg_cell, sample_type='random'):

        if sample_type == 'average':
            pos_choose_cell = y[pos_cell].mean(dim=1)
            pos_dists = self.euclid(y, pos_choose_cell)
        else:
            pos_dists = self.euclid(y, y[pos_cell, :])
        neg_dists = self.euclid(y, y[neg_cell, :])

        # pos_dists = torch.tanh(pos_dists) # 范围[-1, 1] 0->0
        # neg_dists = torch.tanh(neg_dists)

        distance = torch.cat([pos_dists.unsqueeze(dim=-1), neg_dists.unsqueeze(dim=-1)], dim=-1)
        loss = self.CE_loss(distance, torch.ones_like(pos_dists).long())

        return loss


class Cluster_cons_loss(nn.Module):
    def __init__(self, label, device = torch.device('cpu')):
        super(Cluster_cons_loss, self).__init__()
        self.label = label
        #         self.criterion = nn.L1Loss()
        self.label_nums = len(set(self.label))
        self.mask = self.mask_correlated_samples(self.label_nums)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.device = device

    def mask_correlated_samples(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, y, rate=0.3):

        loss_sum = 0
        centers1 = torch.FloatTensor().to(self.device)
        centers2 = torch.FloatTensor().to(self.device)
        for label_id in range(len(set(self.label))):
            y_labels = y[self.label == label_id]

            l_nums = y_labels.shape[0]

            y_center1 = torch.mean(y_labels[torch.randperm(l_nums)[:int(l_nums * rate)]], dim=0)
            y_center2 = torch.mean(y_labels[torch.randperm(l_nums)[:int(l_nums * rate)]], dim=0)

            centers1 = torch.cat([centers1, y_center1.unsqueeze(0)], dim=0)
            centers2 = torch.cat([centers2, y_center2.unsqueeze(0)], dim=0)

        N = 2 * self.label_nums

        c = torch.cat((centers1, centers2), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0))
        sim_i_j = torch.diag(sim, self.label_nums)
        sim_j_i = torch.diag(sim, self.label_nums)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long().to(self.device)
        logits = torch.cat((positive_clusters, negative_clusters), dim=1).to(self.device)
        loss = self.criterion(logits, labels)

        loss /= N

        return loss


def getPosScore(traj):
    scoreorder = 0
    for i in range(len(traj)):
        for j in range(i, len(traj)):
            scoreorder += np.sum((traj[j:] - traj[j]))

    optscoreorder = 0
    sort_traj = np.sort(traj)
    for i in range(len(sort_traj)):
        for j in range(i, len(sort_traj)):
            optscoreorder += np.sum((sort_traj[j:] - sort_traj[j]))
    return scoreorder / optscoreorder