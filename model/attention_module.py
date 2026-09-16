import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):

    def __init__(self, hidden_size):

        super(AttentionModule, self).__init__()

        self.hidden_size = hidden_size

        # global attention
        self.linear_one = nn.Linear(hidden_size, hidden_size)
        self.linear_two = nn.Linear(hidden_size, hidden_size)
        self.linear_three = nn.Linear(hidden_size, 1, bias=False)

        # target aware attention
        self.linear_t = nn.Linear(hidden_size, hidden_size, bias=False)

        # combine interests
        self.linear_combine = nn.Linear(hidden_size * 3, hidden_size)


    def forward(self, hidden, mask, item_embeddings):

        batch_size = hidden.shape[0]

        # Local Interest
        ht = hidden[torch.arange(batch_size), torch.sum(mask,1)-1]

        # Global Attention
        q1 = self.linear_one(ht).view(batch_size,1,self.hidden_size)
        q2 = self.linear_two(hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        alpha = F.softmax(alpha,1)

        s_global = torch.sum(alpha * hidden * mask.view(batch_size,-1,1).float(),1)

        # Target Aware Attention
        hidden_masked = hidden * mask.view(batch_size,-1,1).float()

        qt = self.linear_t(hidden_masked)

        beta = F.softmax(item_embeddings @ qt.transpose(1,2), -1)

        s_target = beta @ hidden_masked

        # Combine
        s_local = ht.unsqueeze(1).expand_as(s_target)
        s_global = s_global.unsqueeze(1).expand_as(s_target)

        s_final = torch.cat([s_target, s_local, s_global], dim=2)

        s_final = self.linear_combine(s_final)

        return s_final