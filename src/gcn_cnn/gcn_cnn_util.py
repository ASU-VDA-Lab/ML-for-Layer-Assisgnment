import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNGraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch= None):
        h = F.relu(self.gcn1(x, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        return self.lin(h)  #[B, out_dim]


class CongestionCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=2, padding=2),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        self.fc = nn.Linear(32, output_dim)  #final embedding

    def forward(self, img):  #img: [B, 6, H, W]
        x = self.encoder(img)
        x = x.view(x.size(0), -1)  #[B, 16]
        return self.fc(x)  #[B, output_dim]
    
class EdgeLayerPredictor(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, cnn_dim, gcn_dim, hidden_dim, num_classes):
        super().__init__()
        self.bias_gain = nn.Parameter(torch.tensor(0.0))

        self.gcn_encoder = GCNGraphEncoder(node_feat_dim, 64, gcn_dim)
        self.cnn_encoder = CongestionCNN(output_dim=cnn_dim)

        self.proj_node = nn.Sequential(nn.Linear(gcn_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.proj_edge = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.proj_cnn  = nn.Sequential(nn.Linear(cnn_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch, net_crops, edge_net_idx):
        #Node embeddings
        h = self.gcn_encoder(x, edge_index, batch)  #[num_nodes, gcn_dim]

        #CNN per-net embeddings, net_crops is [N_nets, C(=6), Ht, Wt]
        cnn_feats_per_net = self.cnn_encoder(net_crops)              #[N_nets, cnn_dim]

        #Map per-net to per-edge using edge_net_idx
        cnn_feats_per_edge = cnn_feats_per_net[edge_net_idx]         #[E_dir, cnn_dim]

        #per-edge inputs
        src, dst = edge_index
        h_src = self.proj_node(h[src])                               #[E_dir, hidden]
        h_dst = self.proj_node(h[dst])                               #[E_dir, hidden]
        edge_attr = self.proj_edge(edge_attr)                        #[E_dir, hidden]
        cnn_feats_per_edge = self.proj_cnn(cnn_feats_per_edge)       #[E_dir, hidden]

        edge_input = torch.cat([h_src, h_dst, edge_attr, cnn_feats_per_edge], dim=1)
        return self.classifier(edge_input)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        y = batch.y.to(device)
        batch_vec = batch.batch.to(device)

        net_crops = batch.net_crops.to(device)            
        edge_net_idx = batch.edge_net_idx.to(device)      

        optimizer.zero_grad()
        out = model(x, edge_index, edge_attr, batch_vec, net_crops, edge_net_idx)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


from sklearn.metrics import accuracy_score, classification_report

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        y = batch.y.to(device)
        batch_vec = batch.batch.to(device)

        net_crops = batch.net_crops.to(device)
        edge_net_idx = batch.edge_net_idx.to(device)

        out = model(x, edge_index, edge_attr, batch_vec, net_crops, edge_net_idx)
        preds = out.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    return acc, report


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        logp = nn.functional.log_softmax(logits, dim=1)
        p = logp.exp()
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)          
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)   
        loss = -(1 - pt) ** self.gamma * logpt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean() if self.reduction == "mean" else loss.sum()



