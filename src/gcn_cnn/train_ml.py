import argparse, sys, os, glob, subprocess, re
import torch, random
from torch_geometric.data import Data
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from collections import Counter
from typing import Optional, Tuple

from helpers import split_datapoint_random, split_explicit_by_design, split_within_design, summarize_split, design_of
from data_struct_util import CoreGrid3D, layer_configs
from gcn_cnn_util import EdgeLayerPredictor
from parser_utils import parse_route_guide_from_file, parse_pin_port_file, parse_fixed_nodes_from_def, merge_fixed_layer_nodes, load_nets_pin_locations
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import torch.nn.functional as F
import helpers
print("helpers imported from:", helpers.__file__)


class RCLogitBiasUrgent(nn.Module):
    def __init__(self,slack_idx=0, r_start=1, c_start=7):
        super().__init__()
        
        
        self.SLACK_IDX = slack_idx
        self.R_START   = r_start   
        self.C_START   = c_start

        #learnable parts
        self.w = nn.Parameter(torch.zeros(7))
        nn.init.uniform_(self.w, -0.05, 0.05)
        self.b_per_class = nn.Parameter(torch.zeros(6))
        self.beta = nn.Parameter(torch.tensor(0.1))   
        self.temp = nn.Parameter(torch.tensor(4.0))   
    def forward(self, eattr_norm: torch.Tensor) -> torch.Tensor:

        slack = eattr_norm[:, self.SLACK_IDX]                                  # [B]
        R     = eattr_norm[:, self.R_START:self.R_START+6]                      # [B,6]
        C     = eattr_norm[:, self.C_START:self.C_START+6]                      # [B,6]
        tau   = R * C                                                  # [B,6]

        urgency = torch.sigmoid(-self.beta * slack).unsqueeze(1)       # [B,1]
        S_eff   = slack.abs().clamp_min(1e-6).unsqueeze(1)             # [B,1]

        #[B,6,7]: [R, C, tau, R/S, C/S, tau/S, 1]
        feats = torch.stack([R, C, tau, R/S_eff, C/S_eff, tau/S_eff,
                             torch.ones_like(R)], dim=-1)               # [B,6,7]

        bias = (feats @ self.w) + self.b_per_class                     # [B,6]
        bias = urgency * bias
        return bias / self.temp.clamp_min(1e-3)


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None):
        if self.ignore_index is not None and self.ignore_index >= 0:
            valid = (target != self.ignore_index)
            if not valid.any():
                return logits.sum() * 0.0
            logits = logits[valid]
            target = target[valid]
            if weight is not None:
                weight = weight[valid]

        logp = F.log_softmax(logits.float(), dim=1)
        p = logp.exp()

        tgt = target.unsqueeze(1)
        pt = p.gather(1, tgt).squeeze(1)       
        logpt = logp.gather(1, tgt).squeeze(1)

        if self.label_smoothing > 0.0:
            s = self.label_smoothing
            ce = -((1.0 - s) * logpt + s * logp.mean(dim=1))
        else:
            ce = -logpt

        focal_factor = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = focal_factor * ce

        if self.alpha is not None:
            loss = loss * self.alpha[target]
        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def class_counts_from_samples(train_samples, C=6) -> torch.Tensor:
    cnt = [0]*C
    for samples, _ in train_samples:
        for s in samples:
            cid = int(s["label"])
            if 0 <= cid < C:
                cnt[cid] += 1
    return torch.tensor(cnt, dtype=torch.long)

def class_balanced_alpha(counts: torch.Tensor, beta: float = 0.9999,
                         normalize: bool = True, clip_range: Optional[Tuple[float,float]] = None,
                         device=None) -> torch.Tensor:
    counts = counts.to(torch.float32)
    counts_clamped = counts.clamp_min(1.0)  
    eff_num = 1.0 - (beta ** counts_clamped)
    alpha = (1.0 - beta) / eff_num  

    if normalize:
        alpha = alpha / alpha.mean().clamp_min(1e-8)
    if clip_range is not None:
        lo, hi = clip_range
        alpha = alpha.clamp(min=lo, max=hi)
    if device is not None:
        alpha = alpha.to(device)
    return alpha

LOSS_ABBR = {"weighted_ce": "wce", "focal": "fl", "weighted_focal": "wfl"}


def pick_free_gpu(min_free_mem_mb=10000):
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,index",
             "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip().split("\n")

        gpus = [(int(r.split(",")[0].strip()), r.split(",")[1].strip()) for r in result]
        gpus = [(mem, idx) for mem, idx in gpus if mem >= min_free_mem_mb]
        if not gpus:
            return None
        best_gpu = max(gpus, key=lambda x: x[0])
        return best_gpu[1]

    except Exception as e:
        print(f"[WARN] Could not query GPUs: {e}")
        return None



os.chdir(os.path.dirname(os.path.abspath(__file__)))
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "asap7"))
pattern = os.path.join(root, "*", "data_point_*")
all_folders = sorted(glob.glob(pattern))

if not all_folders:
    raise FileNotFoundError(f"No folders found for pattern: {pattern}")


all_designs_found = sorted({design_of(p) for p in all_folders})
print("All designs found:", all_designs_found)


def class_to_layer(c: int) -> int:
    return c + 2

def orient_of_layer(layer: int) -> str:
    tup = layer_configs.get(layer, None)
    d = str(tup[0]).upper()
    if d.startswith("H"): return "H"
    if d.startswith("V"): return "V"

def class_mask_from_orient(orient: str, num_classes: int = 6, device=None) -> torch.Tensor:
    mask = torch.ones(num_classes, dtype=torch.bool, device=device)
    if orient not in ("H", "V"):
        return mask
    allowed = []
    for c in range(num_classes):
        L = class_to_layer(c)
        if orient_of_layer(L) == orient:
            allowed.append(c)
    if len(allowed) == 0:
        return mask
    mask[:] = False
    mask[allowed] = True
    return mask


def bbox_in_cells(net, coord_to_cell):
    rows, cols = [], []
    net_nodes = set(edge_l.node1 for edge_l in net.edges_2d_with_layers).union(set(edge_l.node2 for edge_l in net.edges_2d_with_layers))
    if not net_nodes:
        return 0, 0, 0, 0
    for nd in net_nodes:
        r, c = coord_to_cell(nd.coord[0], nd.coord[1])
        rows.append(r); cols.append(c)
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return r0, r1, c0, c1

def build_data_from_shared_nets(net_dict, congestion_img, name_to_idx=None,net_names=None):
    edge_owner = []
    node_map = {}
    edge_samples = []
    all_edges = []

    def flatten_feat(feat):
        if isinstance(feat, list) and any(isinstance(f, list) for f in feat):
            return [item for sublist in feat for item in (sublist if isinstance(sublist, list) else [sublist])]
        return feat

    for net_name, net in net_dict.items():
        for node in net.nodes.values():
            unique_key = f"{net_name}:{node.name}"
            node_map[unique_key] = node
        for e in net.edges_2d_with_layers:
            all_edges.append((net_name, e))

    node_key_list = list(node_map.keys())
    node_list = [node_map[k] for k in node_key_list]
    node_index_map = {k: idx for idx, k in enumerate(node_key_list)}
    x = torch.tensor([node.get_feature_vector() for node in node_list], dtype=torch.float)
    if torch.isnan(x).any() or torch.isinf(x).any():
        bad_cols = torch.where(torch.isnan(x).any(0) | torch.isinf(x).any(0))[0].tolist()
        print(f"[WARN] NaN/Inf in node features; bad columns: {bad_cols}")
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    seen = set()
    edge_index = []
    edge_attr = []

    for net_key, edge in all_edges:
        i = node_index_map[f"{net_key}:{edge.node1.name}"]
        j = node_index_map[f"{net_key}:{edge.node2.name}"]
        key = tuple(sorted([i, j]))
        if key not in seen:
            edge_index.extend([[i, j], [j, i]])
            flat_feat = flatten_feat(edge.get_feature_vector())
            edge_attr.extend([flat_feat, flat_feat])
            if name_to_idx is not None:
                nidx = name_to_idx[net_key]
                edge_owner.extend([nidx, nidx])
            seen.add(key)

    edge_index = torch.tensor(edge_index).T.contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
        bad_cols = torch.where(torch.isnan(edge_attr).any(0) | torch.isinf(edge_attr).any(0))[0].tolist()
        print(f"[WARN] NaN/Inf in global edge_attr; bad columns: {bad_cols}")
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e6, neginf=-1e6)

    batch = torch.zeros(len(x), dtype=torch.long)

    for net_name, net in net_dict.items():
        for edge in net.edges_2d_with_layers:
            src = node_index_map[f"{net_name}:{edge.node1.name}"]
            dst = node_index_map[f"{net_name}:{edge.node2.name}"]
            orient = "H" if edge.edge_direction_val == 0 else "V"
            edge_feat = torch.tensor(flatten_feat(edge.get_feature_vector()), dtype=torch.float)
            edge_feat = torch.nan_to_num(edge_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            sample = {
                "src": src,
                "dst": dst,
                "src_name": edge.node1.name,
                "dst_name": edge.node2.name,
                "net_key": net_name,
                "edge_attr": edge_feat,
                "label": edge.actual_layer - 2,
                "orient" : orient
            }
            edge_samples.append(sample)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    data.congestion_img = congestion_img
    if name_to_idx is not None:
        data.edge_net_idx = torch.tensor(edge_owner, dtype=torch.long)  
        data.net_names    = net_names                                   
    return data, edge_samples

import torch.nn.functional as F

        
def load_graphs_from_folders(folders, keep_netdict: bool = True):
    graph_list = []
    sample_list = []

    for folder in folders:
        print(f"i am in folder named : {folder}")
        guide = os.path.join(folder, "timing.guide")
        def_file = os.path.join(folder, "5_1_grt.def")
        pin_cap_slacks = os.path.join(folder, "pin_cap_slacks.txt")
        pin_locs = os.path.join(folder, "pin_locations.txt")

        route_edges, fixed_nodes_m1 = parse_route_guide_from_file(guide)
        die_x, die_y, gcell_step_do_values, fixed_nodes_pinfile = parse_fixed_nodes_from_def(def_file)
        merged_fixed_nodes = merge_fixed_layer_nodes(fixed_nodes_m1, fixed_nodes_pinfile)
        net_dict_3d_without_edges = load_nets_pin_locations(pin_locs)
        net_dict_3d, fixed_ports_net_dict, tech_node = parse_pin_port_file(
            pin_cap_slacks, net_dict_3d_without_edges, route_edges, merged_fixed_nodes
        )

        grid_x = gcell_step_do_values[0][1]
        grid_y = gcell_step_do_values[1][1]
        grid = CoreGrid3D(die_x, die_y, grid_x, grid_y, num_layers=7)
        for net3d in net_dict_3d.values():
            grid.update_usage(net3d)
        grid.update_congestion()
        
        net_names   = list(net_dict_3d.keys())
        name_to_idx = {n: i for i, n in enumerate(net_names)}
        graph_data, edge_samples = build_data_from_shared_nets(net_dict_3d, grid.congestion_image, name_to_idx=name_to_idx, net_names=net_names)

        
        graph_data.congestion_img = grid.congestion_image.to(dtype=torch.float32)
        L, H, W = grid.congestion_image.shape  #[6, H, W]
        PAD = 1
        net_names = list(net_dict_3d.keys())
        boxes_rc = []  
        _cell_cache = {}
        def cell_of(nd):
            k = id(nd)
            if k not in _cell_cache:
                _cell_cache[k] = grid.coord_to_cell(nd.coord[0], nd.coord[1])
            return _cell_cache[k]
        
        def bbox_in_cells_fast(net):
            rs, cs = [], []
            for nd in net.nodes.values():
                r, c = cell_of(nd)
                rs.append(r); cs.append(c)
            return min(rs), max(rs), min(cs), max(cs)
        
        for net_name in net_names:
            net = net_dict_3d[net_name]
            r0, r1, c0, c1 = bbox_in_cells_fast(net)
            r0 = max(0, r0 - PAD); r1 = min(H-1, r1 + PAD)
            c0 = max(0, c0 - PAD); c1 = min(W-1, c1 + PAD)
            boxes_rc.append((r0, c0, r1, c1))
        
        graph_data.net_names = net_names
        graph_data.net_boxes_rc = torch.tensor(boxes_rc, dtype=torch.int16)

        if not edge_samples:
            print(f"[WARN] datapoint has 0 edge samples: {folder}", flush=True)
            raise ValueError(f"check {folder} as it doesnt have any edge sample")

        print(f"[LOAD]   nodes={graph_data.x.size(0)} edges(attr)={graph_data.edge_attr.size(0)} samples={len(edge_samples)}",
              flush=True)

        graph_list.append(graph_data)
        sample_list.append((edge_samples, net_dict_3d if keep_netdict else None))

        if not keep_netdict:
            net_dict_3d = None

    return graph_list, sample_list



DEFAULT_NORM_PATH = "norm_stats.pt"

@torch.no_grad()
def compute_and_save_normalization(train_graphs, save_path: str = DEFAULT_NORM_PATH):
    node_feats, edge_feats = [], []
    img_sum = None
    img_sqsum = None
    img_count = 0
    
    for data in train_graphs:
        if getattr(data, "x", None) is not None and data.x.numel() > 0:
            node_feats.append(data.x.detach().float().cpu())
        if getattr(data, "edge_attr", None) is not None and data.edge_attr.numel() > 0:
            edge_feats.append(data.edge_attr.detach().float().cpu())
         
        img = getattr(data, "congestion_img", None)
        if torch.is_tensor(img) and img.numel() > 0:
            img = torch.nan_to_num(img.detach().float().cpu())
            if img_sum is None:
                C = img.shape[0]
                img_sum   = torch.zeros(C, dtype=torch.float32)
                img_sqsum = torch.zeros(C, dtype=torch.float32)
            img_sum   += img.sum(dim=(1, 2))
            img_sqsum += (img * img).sum(dim=(1, 2))
            img_count += img.shape[1] * img.shape[2]
    if len(node_feats) == 0:
        raise ValueError("No node features in training graphs.")
    if len(edge_feats) == 0:
        raise ValueError("No edge features in training graphs.")



    for i in range(len(node_feats)):
        node_feats[i] = torch.nan_to_num(node_feats[i], nan=0.0, posinf=1e6, neginf=-1e6)
    for i in range(len(edge_feats)):
        edge_feats[i] = torch.nan_to_num(edge_feats[i], nan=0.0, posinf=1e6, neginf=-1e6)

    all_nodes = torch.cat(node_feats, dim=0)
    all_edges = torch.cat(edge_feats, dim=0)

    node_mean = all_nodes.mean(dim=0)
    node_std  = all_nodes.std(dim=0, unbiased=False).clamp_min(1e-8)
    edge_mean = all_edges.mean(dim=0)
    edge_std  = all_edges.std(dim=0, unbiased=False).clamp_min(1e-8)
    
    
    stats = {
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
    }
    if img_count > 0:
        img_mean = (img_sum / img_count)
        img_var  = (img_sqsum / img_count) - img_mean * img_mean
        img_std  = torch.sqrt(img_var.clamp_min(1e-8))
        stats["img_mean"] = img_mean
        stats["img_std"]  = img_std
        
    torch.save(stats, save_path)
    return stats

def load_norm_stats(path: str = DEFAULT_NORM_PATH, map_location=None):
    stats = torch.load(path, map_location=map_location or "cpu")
    for k in ("node_mean", "node_std", "edge_mean", "edge_std", "img_mean", "img_std"):
        if not torch.is_tensor(stats[k]): stats[k] = torch.tensor(stats[k], dtype=torch.float32)

    return stats


def normalize_graph_inplace(data, edge_samples, stats, device=None):
    node_mean = stats["node_mean"].to(device) if device else stats["node_mean"]
    node_std  = stats["node_std"].to(device)  if device else stats["node_std"]
    edge_mean = stats["edge_mean"].to(device) if device else stats["edge_mean"]
    edge_std  = stats["edge_std"].to(device)  if device else stats["edge_std"]


    if getattr(data, "x", None) is not None and data.x.numel() > 0:
        data.x = (data.x - node_mean) / node_std
        data.x = torch.nan_to_num(data.x)
    if getattr(data, "edge_attr", None) is not None and data.edge_attr.numel() > 0:
        data.edge_attr = (data.edge_attr - edge_mean) / edge_std
        data.edge_attr = torch.nan_to_num(data.edge_attr)

    if edge_samples:
        for s in edge_samples:
            e = s.get("edge_attr", None)
            if torch.is_tensor(e):
                s["edge_attr"] = torch.nan_to_num((e.to(edge_mean.device) - edge_mean) / edge_std)
    if hasattr(data, "congestion_img") and torch.is_tensor(data.congestion_img):
        m = stats["img_mean"].to(device) if device else stats["img_mean"]
        s = stats["img_std"].to(device)  if device else stats["img_std"]
        data.congestion_img = torch.nan_to_num((data.congestion_img - m[:, None, None]) / s[:, None, None])
    
    return data, edge_samples

def prep_edge_tensors_for_graph(samples, device):
    if len(samples) == 0:
        return None
    src_idx = torch.tensor([s["src"] for s in samples], dtype=torch.long, device=device)
    dst_idx = torch.tensor([s["dst"] for s in samples], dtype=torch.long, device=device)
    edge_attr = torch.stack([s["edge_attr"] for s in samples]).to(device)
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.long, device=device)
    return {"src": src_idx, "dst": dst_idx, "edge_attr": edge_attr, "labels": labels}



def build_net_cnn_feats(model, data, device, out_hw=(16,16)):
    img = data.congestion_img.unsqueeze(0).to(device) 
    boxes_rc = data.net_boxes_rc.to(device=device, dtype=torch.int32)

    boxes_xy = torch.stack([
        boxes_rc[:,1].float(),            
        boxes_rc[:,0].float(),            
        (boxes_rc[:,3].float()+1.0),      
        (boxes_rc[:,2].float()+1.0),      
    ], dim=1)


    from torchvision.ops import roi_align
    rois = [boxes_xy]
    crops = roi_align(
        img, rois,
        output_size=out_hw,
        spatial_scale=1.0,
        aligned=True,
    )  


    v = model.cnn_encoder(crops)      
    v = model.proj_cnn(v)             
    return v
    
@torch.no_grad()
def eval_epoch(model, graph_data_list, sample_list, criterion, device):
    model.eval()
    total_loss_sum, total_correct, total_samples = 0.0, 0, 0

    for data, (samples, _) in zip(graph_data_list, sample_list):
        gcn_out = model.gcn_encoder(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        net_vec_table = build_net_cnn_feats(model, data, device)          
        name_to_idx = {n: i for i, n in enumerate(data.net_names)}
        net_idx_per_sample = torch.tensor([name_to_idx[s["net_key"]] for s in samples],dtype=torch.long, device=device)

        if len(samples) == 0:
            continue

        et = prep_edge_tensors_for_graph(samples, device)
        src, dst, eattr, labels_all = et["src"], et["dst"], et["edge_attr"], et["labels"]
        orients = [s["orient"] for s in samples]

        N  = labels_all.numel()
        bs = _auto_edge_batch_size(device, N)
        start = 0
        while start < N:
            end = min(start + bs, N)
            sl  = slice(start, end)

            h1 = model.proj_node(gcn_out[src[sl]])
            h2 = model.proj_node(gcn_out[dst[sl]])
            ea = model.proj_edge(eattr[sl])
            cnn_per_edge = net_vec_table[net_idx_per_sample[sl]]

            feats = torch.cat([h1, h2, ea, cnn_per_edge], dim=1)


            out = model.classifier(feats)
            rcb = model.rc_bias_mod(eattr[sl])
            rcb = rcb - rcb.mean(dim=1, keepdim=True)
            out = out + model.bias_gain * rcb
            
            masks = torch.stack([class_mask_from_orient(o, num_classes=out.size(1), device=out.device)
                                 for o in orients[start:end]])
            out = out.masked_fill(~masks, -1e9)

            lb  = labels_all[sl]
            loss = criterion(out, lb)

            bsz = lb.numel()
            total_loss_sum += loss.item() * bsz
            total_correct  += (out.argmax(1) == lb).sum().item()
            total_samples  += bsz
            start = end

    if total_samples == 0:
        return float("nan"), float("nan")
    return total_loss_sum / total_samples, total_correct / total_samples





def train_epoch(model, graph_data_list, sample_list, optimizer, criterion, device, epoch=None, folders=None):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    combined = list(zip(graph_data_list, sample_list, folders or [None]*len(graph_data_list)))
    random.shuffle(combined)
    graph_data_list, sample_list, folders = zip(*combined)

    for gi, (data, (samples, _)) in enumerate(zip(graph_data_list, sample_list)):
        if folders is not None:
            print(f"[TRAIN][Epoch {epoch}] Datapoint: {folders[gi]}", flush=True)

        gcn_out = model.gcn_encoder(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        
        net_vec_table = build_net_cnn_feats(model, data, device)          
        name_to_idx = {n: i for i, n in enumerate(data.net_names)}
        net_idx_per_sample = torch.tensor([name_to_idx[s["net_key"]] for s in samples],dtype=torch.long, device=device)

        if len(samples) == 0:
            continue

        et = prep_edge_tensors_for_graph(samples, device)
        src, dst, eattr, labels_all = et["src"], et["dst"], et["edge_attr"], et["labels"]
        orients = [s["orient"] for s in samples]

        optimizer.zero_grad(set_to_none=True)

        N  = labels_all.numel()
        bs = _auto_edge_batch_size(device, N)
        start = 0

        chunk_losses = []
        graph_correct = 0
        graph_samples = 0

        while start < N:
            end = min(start + bs, N)
            sl  = slice(start, end)
            try:
                h1 = model.proj_node(gcn_out[src[sl]])
                h2 = model.proj_node(gcn_out[dst[sl]])
                ea = model.proj_edge(eattr[sl])

                cnn_per_edge = net_vec_table[net_idx_per_sample[sl]]

                feats = torch.cat([h1, h2, ea, cnn_per_edge], dim=1)

                out = model.classifier(feats)
                rcb = model.rc_bias_mod(eattr[sl])
                rcb = rcb - rcb.mean(dim=1, keepdim=True)
                out = out + model.bias_gain * rcb
                
                masks = torch.stack([class_mask_from_orient(o, num_classes=out.size(1), device=out.device)
                                     for o in orients[start:end]])
                out_masked = out.masked_fill(~masks, -1e9)

                lb = labels_all[sl]
                loss = criterion(out_masked, lb)
                chunk_losses.append(loss)

                pred = out_masked.argmax(1)
                graph_correct += (pred == lb).sum().item()
                graph_samples += lb.numel()

                start = end
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    bs_before = bs
                    if bs <= 4096:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    bs = max(4096, bs // 2)
                    print(f"[TRAIN][Epoch {epoch}] OOM -> reducing edge batch {bs_before} -> {bs}", flush=True)
                else:
                    raise

        if chunk_losses:
            graph_loss = torch.stack(chunk_losses).mean()
            graph_loss.backward()
            optimizer.step()

            total_loss    += graph_loss.item()
            total_correct += graph_correct
            total_samples += graph_samples

    avg_loss = total_loss / max(1, len(graph_data_list))
    acc      = total_correct / max(1, total_samples)
    return avg_loss, acc

def build_edge_lookup(net_dict):
    lookup = {}
    for net_name, net in net_dict.items():
        for edge in net.edges_2d_with_layers:
            key = (net_name, tuple(sorted([edge.node1.name, edge.node2.name])))
            lookup[key] = edge
    return lookup

@torch.no_grad()
def test_and_assign(model, graph_data_list, sample_list, device, batch_size=65536):
    model.eval()
    all_preds, all_labels = [], []

    for data, (samples, net_dict) in zip(graph_data_list, sample_list):
        gcn_out = model.gcn_encoder(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        net_vec_table = build_net_cnn_feats(model, data, device)          
        
        name_to_idx = {n: i for i, n in enumerate(data.net_names)}
        net_idx_per_sample = torch.tensor([name_to_idx[s["net_key"]] for s in samples],dtype=torch.long, device=device)

        edge_lookup = build_edge_lookup(net_dict)

        if len(samples) == 0:
            continue

        et = prep_edge_tensors_for_graph(samples, device)
        src, dst, eattr, labels_all = et["src"], et["dst"], et["edge_attr"], et["labels"]
        orients = [s["orient"] for s in samples]
        net_keys = [s["net_key"] for s in samples]
        src_names = [s["src_name"] for s in samples]
        dst_names = [s["dst_name"] for s in samples]

        N = labels_all.numel()
        start = 0
        while start < N:
            end = min(start + batch_size, N)
            sl  = slice(start, end)

            h1 = model.proj_node(gcn_out[src[sl]])
            h2 = model.proj_node(gcn_out[dst[sl]])
            ea = model.proj_edge(eattr[sl])
            
            cnn_per_edge = net_vec_table[net_idx_per_sample[sl]]
            feats = torch.cat([h1, h2, ea, cnn_per_edge], dim=1)

            out = model.classifier(feats)
            rcb = model.rc_bias_mod(eattr[sl])
            rcb = rcb - rcb.mean(dim=1, keepdim=True)
            out = out + model.bias_gain * rcb
            
            masks = torch.stack([class_mask_from_orient(o, num_classes=out.size(1), device=out.device)
                                 for o in orients[start:end]])
            out   = out.masked_fill(~masks, -1e9)
            preds = out.argmax(1).cpu()

            for i, pred in enumerate(preds.tolist()):
                predicted_layer = pred + 2
                key = (net_keys[start + i], tuple(sorted([src_names[start + i], dst_names[start + i]])))
                if key in edge_lookup:
                    edge_lookup[key].predicted_layer = predicted_layer
                all_preds.append(predicted_layer)

            all_labels.extend((labels_all[sl] + 2).tolist())
            start = end

    print(Counter(all_labels))
    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, labels=[2,3,4,5,6,7], digits=4))
    print("Macro F1:", f1_score(all_labels, all_preds, labels=[2,3,4,5,6,7], average="macro"))
    return all_labels, all_preds


def make_weighted_ce(train_samples, device, C=6):
    from collections import Counter
    cnt = Counter()
    for samples,_ in train_samples:
        cnt.update([s["label"] for s in samples])
    n = sum(cnt.values())
    freq = torch.tensor([cnt.get(i,0) for i in range(C)], dtype=torch.float)

    freq = freq + 1.0
    w = (n / freq)
    w = w / w.mean()
    return torch.nn.CrossEntropyLoss(weight=w.to(device)), freq

def _auto_edge_batch_size(device, n_edges: int) -> int:
    if n_edges <= 0:
        return 0
    if torch.cuda.is_available() and device.type == "cuda":
        if n_edges >= 800_000:
            return 131072
        elif n_edges >= 200_000:
            return 65536
        else:
            return 32768
    else:
        if n_edges >= 200_000:
            return 16384
        else:
            return 8192

def simple_val_from_train(train_folders, val_size=0.1, random_state=42):
    n = len(train_folders)
    if val_size <= 0 or n < 2:
        print(f"[VAL] Skipping validation: val_size={val_size}, train_folders={n}")
        return train_folders, []
    n_val = max(1, int(round(val_size * n)))
    if n_val >= n:
        n_val = n - 1
    rng = np.random.RandomState(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_idx = set(idx[:n_val])
    val_folders   = [train_folders[i] for i in val_idx]
    train_folders = [train_folders[i] for i in idx[n_val:]]
    print(f"[VAL] folder-level split: train={len(train_folders)}  val={len(val_folders)}")
    return train_folders, val_folders

def designs_of(folders):
    return sorted({design_of(p) for p in folders}) if folders else []

def datapoint_id_of(path: str) -> Optional[int]:
    m = re.search(r"data_point_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None

def parse_id_list(tokens: Optional[list]) -> set:
    ids = set()
    if not tokens:
        return ids
    for tok in tokens:
        for part in str(tok).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                if a > b: a, b = b, a
                ids.update(range(a, b + 1))
            else:
                ids.add(int(part))
    return ids

def summarize_layer_hist(sample_list, name="TRAIN", save_png=True, out_dir="outputs"):
    labels = [s["label"] for samples, _ in sample_list for s in samples]
    if len(labels) == 0:
        print(f"[HIST] {name}: no samples found.")
        return
    layers = [l + 2 for l in labels]
    cnt = Counter(layers)
    total = sum(cnt.values())
    layer_keys = sorted(cnt.keys())
    counts = [cnt[k] for k in layer_keys]
    perc = [100.0 * c / total for c in counts]
    print(f"\n[HIST] {name} layer distribution (layers 2..9 expected)")
    for k, c, p in zip(layer_keys, counts, perc):
        print(f"  Layer {k:>2}  |  count={c:>6}  ({p:6.2f}%)")
    print(f"  TOTAL         |  count={total:>6} (100.00%)")

    class_cnt = Counter(labels)
    class_keys = sorted(class_cnt.keys())
    print(f"\n[HIST] {name} class IDs (0..7) and corresponding layers:")
    for cid in class_keys:
        L = cid + 2
        print(f"  class {cid} -> layer {L} : count={class_cnt[cid]}")

    if save_png:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.bar([str(k) for k in layer_keys], counts)
        plt.xlabel("Routing Layer ID")
        plt.ylabel("Count")
        plt.title(f"{name} Layer Histogram")
        for i, v in enumerate(counts):
            plt.text(i, v, str(v), ha="center", va="bottom", fontsize=8, rotation=0)
        png_path = os.path.join(out_dir, f"{name.lower()}_layer_histogram.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"[HIST] Saved {name} layer histogram to {png_path}")

def _scenario_tag(args) -> str:
    """
    opt1: explicit mode (non-overlapping designs)
    opt2: cross-design with different datapoints for chosen designs (no leakage) or random cross-design
    opt3: cross-design with edge leakage from explicit test datapoints (test-dps-train-frac > 0)
    """
    if args.mode == "explicit":
        return "opt1"
    if args.mode == "cross-design":
        if args.test_designs and args.test_dps:
            frac = float(getattr(args, "test_dps_train_frac", 0.0) or 0.0)
            return "opt3" if frac > 0.0 else "opt2"
        return "opt2"
    return "one"

def main():
    os.makedirs("outputs", exist_ok=True)
    parser = argparse.ArgumentParser(description="Dataset splitting options")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["one-design", "cross-design", "explicit"],
                        help="Split mode: one-design | cross-design | explicit")

    #Mode 1: within-design
    parser.add_argument("--design", type=str, help="Design name to split within (used only if one-design)")

    #Mode 2: cross-design
    parser.add_argument("--cross-designs", type=str, nargs="+", help="List of designs to consider (mode=cross-design)")
    parser.add_argument("--test-designs", type=str, nargs="+",
                        help="Subset of designs (from --cross-designs) whose datapoints can go to TEST.")
    parser.add_argument("--test-dps", type=str, nargs="+",
                        help="Explicit datapoint IDs within the --test-designs to force into TEST (e.g., '2 3' or '2-4,7').")
    parser.add_argument("--test-dps-train-frac", type=float, default=0.0,
                        help="Fraction [0..1] of EDGE SAMPLES from the explicit TEST datapoints to ALSO include in TRAIN.")

    #Mode 3: explicit
    parser.add_argument("--train-designs", type=str, nargs="+", help="Designs to use for training (mode=explicit)")
    parser.add_argument("--test-designs-explicit", type=str, nargs="+", help="Designs to use for testing (mode=explicit)")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    #common options
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of datapoints to reserve for testing (when applicable)")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of *train* datapoints for validation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--no-val", action="store_true", help="Disable validation split")
    parser.add_argument("--gpu", type=str, default=None, help="GPU id to use (e.g. '0' or '1,2').")

    #Loss selection
    parser.add_argument("--loss", type=str, default="weighted_ce", choices=["weighted_ce", "focal", "weighted_focal"])
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--cb-beta", type=float, default=0.9999,
                        help="Class-Balanced beta for effective number weights (used when --loss weighted_focal).")

    args = parser.parse_args()
    num_epochs = args.epochs

    #sanity checks depending on mode
    if args.mode == "one-design" and not args.design:
        parser.error("--design must be provided when mode=one-design")
    if args.mode == "cross-design" and not args.cross_designs:
        parser.error("--cross-designs must be provided when mode=cross-design")
    if args.mode == "explicit" and (not args.train_designs or not args.test_designs_explicit):
        parser.error("--train-designs and --test-designs-explicit are required when mode=explicit")

    print("ARGS:", vars(args), flush=True)

    #Train/Test split based on mode
    share_edge_frac = {}

    if args.mode == "one-design":
        design_name = args.design.strip()
        design_folders = [p for p in all_folders if design_of(p) == design_name]
        if not design_folders:
            raise ValueError(f"No datapoints found for design: {design_name}")

        train_folders, test_folders = split_within_design(
            design_folders, design=design_name, test_size=args.test_size, random_state=args.random_state
        )

    elif args.mode == "cross-design":
        selected_folders = [p for p in all_folders if design_of(p) in args.cross_designs]
        if not selected_folders:
            raise ValueError(f"No datapoints found for designs: {args.cross_designs}")

        if args.test_designs and args.test_dps:
            unknown = sorted(set(args.test_designs) - set(args.cross_designs))
            if unknown:
                raise ValueError(f"--test-designs includes designs not in --cross-designs: {unknown}")

            test_designs_set = set(args.test_designs)
            test_ids = parse_id_list(args.test_dps)

            train_folders, test_folders = [], []
            missing_ids = set()

            for p in selected_folders:
                d = design_of(p)
                dp_id = datapoint_id_of(p)
                if d in test_designs_set and dp_id is not None and dp_id in test_ids:
                    test_folders.append(p)
                else:
                    train_folders.append(p)

            present_ids = {datapoint_id_of(p) for p in selected_folders if design_of(p) in test_designs_set}
            for tid in test_ids:
                if tid not in present_ids:
                    missing_ids.add(tid)
            if missing_ids:
                print(f"[WARN] Some requested test datapoint IDs not found in {sorted(test_designs_set)}: {sorted(missing_ids)}")

            share_frac = max(0.0, min(1.0, float(args.test_dps_train_frac)))
            if share_frac > 0.0:
                for p in list(test_folders):
                    if p not in train_folders:
                        train_folders.append(p)
                    share_edge_frac[p] = share_frac
                print(f"[SPLIT] Edge-level share enabled for {len(share_edge_frac)} test folders (train frac={share_frac:.3f}).")

            print("[SPLIT][cross-design explicit IDs]")
            print("  Train-only designs:", sorted({design_of(p) for p in train_folders if design_of(p) not in test_designs_set}))
            print("  Test-designs (explicit IDs):", sorted(test_designs_set))
            print(f"  Train datapoints: {len(train_folders)} | Test datapoints: {len(test_folders)}")

        else:
            train_folders, test_folders = split_datapoint_random(
                selected_folders, test_size=args.test_size, random_state=args.random_state
            )

    elif args.mode == "explicit":
        train_folders, test_folders = split_explicit_by_design(
            all_folders,
            train_designs=args.train_designs,
            test_designs=args.test_designs_explicit,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    #Validation split (from train)
    if not args.no_val and args.val_size > 0:
    # Keep any edge-sharing test folders OUT of VAL
        protected = set(share_edge_frac.keys())
        pool = [p for p in train_folders if p not in protected]

        pool_train, val_folders = simple_val_from_train(pool, val_size=args.val_size, random_state=args.random_state)

    #Recombine: protected stay in TRAIN, VAL drawn only from pool
        train_folders = pool_train + [p for p in train_folders if p in protected]
        print(f"[VAL] (protected={len(protected)}) train={len(train_folders)}  val={len(val_folders)}")

    #Sanity: protected must be in TRAIN
        for f in protected:
            assert f in train_folders, f"[BUG] Protected folder slipped out of TRAIN: {f}"
    else:
        val_folders = []
    train_designs = designs_of(train_folders)
    val_designs   = designs_of(val_folders)
    test_designs  = designs_of(test_folders)


    loss_tag = LOSS_ABBR.get(args.loss, "wce")
    scenario_tag = _scenario_tag(args)
    run_id = (
        f"{scenario_tag}"
        f"_test-{'_'.join(test_designs) or 'none'}"
        f"_e{num_epochs}"
        f"_{loss_tag}"
    )

    log_path = os.path.join("outputs", f"log_{run_id}.txt")
    sys.stdout = open(log_path, "w")
    print(f"[INFO] Logging to {log_path}", flush=True)

    summarize_split(train_folders, test_folders, val_folders)
    if val_folders:
        print(f"[INFO] validation datapoints: {len(val_folders)}", flush=True)


    train_graphs, train_samples = load_graphs_from_folders(train_folders, keep_netdict=False)
    val_graphs,   val_samples   = load_graphs_from_folders(val_folders,   keep_netdict=False) if val_folders else ([], [])
    test_graphs,  test_samples  = load_graphs_from_folders(test_folders,  keep_netdict=True)

        
    from collections import Counter
    cnt = Counter()
    for s,_ in train_samples:
        cnt.update([x["label"] for x in s])
        print("Train label counts (0 to 5 i.e M2 TO M7):", cnt)


    #If edge-level sharing is enabled, repartition edges inside explicit test datapoints
    if share_edge_frac:
        print(f"[SPLIT] Re-partitioning edge samples for shared test datapoints into train ({len(share_edge_frac)} folders).")
        rng = np.random.RandomState(args.random_state)
        train_idx = {f: i for i, f in enumerate(train_folders)}
        test_idx  = {f: i for i, f in enumerate(test_folders)}

        for f, frac in share_edge_frac.items():
            if f not in train_idx or f not in test_idx:
                print(f"[WARN] Shared folder missing from one split (train/test): {f}")
                continue

            ti = test_idx[f]
            tri = train_idx[f]

            test_edge_samples, test_net = test_samples[ti]
            n = len(test_edge_samples)
            if n == 0 or frac <= 0.0:
                print(f"[SPLIT] Skipping {f} (n={n}, frac={frac})")
                continue

            k = int(round(frac * n))
            if k <= 0:
                print(f"[SPLIT] k=0 for {f} at frac={frac:.3f} (n={n})")
                continue
            if k >= n:
                k = n

            perm = np.arange(n)
            rng.shuffle(perm)
            take = set(perm[:k])       
            keep = [i for i in perm[k:]]  

            moved_edges  = [test_edge_samples[i] for i in take]
            remain_edges = [test_edge_samples[i] for i in keep]

            train_samples[tri] = (moved_edges, train_samples[tri][1])
            test_samples[ti] = (remain_edges, test_net)

            print(f"[SPLIT] {os.path.basename(f)}: edges total={n}, to-train={len(moved_edges)}, to-test={len(remain_edges)}")


    summarize_layer_hist(train_samples, name="TRAIN", save_png=False, out_dir="outputs")

    #Normalize: fit on TRAIN ONLY, then apply to all splits
    norm_path = os.path.join("outputs", f"norm_{run_id}.pt")
    compute_and_save_normalization(train_graphs, save_path=norm_path)
    stats = load_norm_stats(norm_path)

    for i, d in enumerate(train_graphs):
        normalize_graph_inplace(d, train_samples[i][0], stats)
    for i, d in enumerate(val_graphs):
        if val_graphs:
            normalize_graph_inplace(d, val_samples[i][0], stats)
    for i, d in enumerate(test_graphs):
        normalize_graph_inplace(d, test_samples[i][0], stats)

    with torch.no_grad():
        x_all = torch.cat([g.x for g in train_graphs], dim=0)
        e_all = torch.cat([g.edge_attr for g in train_graphs if g.edge_attr.numel() > 0], dim=0)
        print("Train nodes mean/std ~", x_all.mean(0), x_all.std(0))
        print("Train edges mean/std ~", e_all.mean(0), e_all.std(0))


    env_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"[INFO] Using user-specified GPU(s): {args.gpu}")
    elif env_visible:
        print(f"[INFO] Respecting pre-set CUDA_VISIBLE_DEVICES={env_visible}")
    else:
        free_gpu = pick_free_gpu(min_free_mem_mb=15360)
        if free_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu
            print(f"[INFO] Using auto-selected GPU {free_gpu} with sufficient free memory.")
        else:
            print("[WARN] No suitable GPU found. Falling back to CPU.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        logical = torch.cuda.current_device()
        name = torch.cuda.get_device_name(logical)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        physical = None
        if visible:
            try:
                physical = visible.split(",")[logical].strip()
            except Exception:
                pass
        if physical is not None:
            print(f"[GPU] Using logical CUDA:{logical} -> physical GPU {physical}: {name}", flush=True)
        else:
            print(f"[GPU] Using logical CUDA:{logical}: {name}", flush=True)

    model = EdgeLayerPredictor(
        node_feat_dim=5,
        edge_feat_dim=13,
        cnn_dim=64,
        gcn_dim=64,
        hidden_dim=128,
        num_classes=6
    ).to(device)
    model.rc_bias_mod = RCLogitBiasUrgent(slack_idx=0, r_start=1, c_start=7).to(device)


    _, freq = make_weighted_ce(train_samples, device)
    print("class priors (smoothed):", (freq / freq.sum()).tolist())
    with torch.no_grad():
        priors = (freq / freq.sum()).clamp_(1e-6, 1.0).to(device)
        model.classifier[-1].bias.copy_(priors.log())

    if args.loss == "weighted_ce":
        criterion, _ = make_weighted_ce(train_samples, device)
        print("[LOSS] Using WEIGHTED CrossEntropy")
        print("class weights:", criterion.weight.detach().cpu().tolist())

    elif args.loss == "focal":
        criterion = FocalLoss(alpha=None, gamma=args.gamma)
        print("[LOSS] Using FocalLoss (no alpha)")
        print(f"[LOSS] gamma={args.gamma}")

    else:
        counts = class_counts_from_samples(train_samples, C=6)
        beta = args.cb_beta
        alpha = class_balanced_alpha(counts, beta=beta, normalize=True, clip_range=(0.1, 2.0), device=device)
        criterion = FocalLoss(alpha=alpha, gamma=args.gamma)
        print("[LOSS] Using WEIGHTED FocalLoss (Class-Balanced alpha)")
        print(f"[LOSS] cb_beta={beta}  gamma={args.gamma}")
        print("alpha (per-class):", alpha.detach().cpu().tolist())
        print("counts (per-class):", counts.detach().cpu().tolist())
    

    optimizer = torch.optim.Adam(list(model.parameters()),lr=3e-4, weight_decay=1e-4)

    best_val = float("inf")
    patience = 10
    stale = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_graphs, train_samples, optimizer, criterion, device, epoch=epoch, folders=train_folders)
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        if val_graphs:
            val_loss, val_acc = eval_epoch(model, val_graphs, val_samples, criterion, device)
            print(f"Epoch {epoch:02d} | Train: loss={train_loss:.4f} acc={train_acc:.4f} "
                  f"| Val:  loss={val_loss:.4f} acc={val_acc:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"[EARLY STOP] no val improvement for {patience} epochs.", flush=True)
                    break

    actual_layers, predicted_layers = test_and_assign(model, test_graphs, test_samples, device)
    conf_mat = confusion_matrix(actual_layers, predicted_layers, labels=[2,3,4,5,6,7])
    cm_percent = conf_mat.astype(float) / conf_mat.sum(axis=1)[:, np.newaxis] * 100

    annot_labels = np.array([["{:.1f}%".format(val) for val in row] for row in cm_percent])


    plt.figure(figsize=(8,6))
    ax = sns.heatmap(
        conf_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[2,3,4,5,6,7],
        yticklabels=[2,3,4,5,6,7],
        annot_kws={"size": 14}   
    )

    plt.xlabel('Predicted Layer', fontsize=18)
    plt.ylabel('Actual Layer', fontsize=18)
    plt.title('Confusion Matrix (Layer Prediction)', fontsize=20)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Counts", fontsize=16)
    plt.tight_layout()
    conf_mat_path = os.path.join("outputs", f"confusion_matrix_{run_id}.png")
    plt.savefig(conf_mat_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    ax = sns.heatmap(
        cm_percent,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        xticklabels=[2,3,4,5,6,7],
        yticklabels=[2,3,4,5,6,7],
        annot_kws={"size": 15},   
        cbar=True,
        cbar_kws={"shrink": 0.9, "label": "Percentage (%)"},
        square=True
    )


    plt.xlabel('Predicted Layer', fontsize=15)
    plt.ylabel('Actual Layer', fontsize=15)
    plt.title('Confusion Matrix (Layer Prediction in Percentage)', fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Percentage (%)", fontsize=15)
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(10))  

    plt.tight_layout()
    conf_mat_path = os.path.join("outputs", f"confusion_matrix_percent_{run_id}.png")
    plt.savefig(conf_mat_path, dpi=300)
    print(f"[INFO] Saved confusion matrix to {conf_mat_path}")
    plt.close()

 
    #plt.figure(figsize=(8,6))
    #sns.heatmap(cm_percent, annot=annot_labels, fmt="", cmap="Blues", vmin=0, vmax=100, xticklabels=[2,3,4,5,6,7], yticklabels=[2,3,4,5,6,7],annot_kws={"size":15}, cbar=True,cbar_kws={"shrink": 0.9, "label": "Percentage (%)"},square=True)
    #plt.xlabel('Predicted Layer',fontsize=15)
    #plt.ylabel('Actual Layer',fontsize=15)
    #plt.title('Confusion Matrix (Layer Prediction in Percentage)',fontsize=15)
    #plt.tight_layout()
    #conf_mat_path = os.path.join("outputs", f"confusion_matrix_percent_{run_id}.png")
    #plt.savefig(conf_mat_path, dpi=300)
    #print(f"[INFO] Saved confusion matrix to {conf_mat_path}")
    #plt.close()

    
    plt.figure(figsize=(8,6))
    plt.scatter(actual_layers, predicted_layers, alpha=0.6)
    plt.xlabel("Actual Layer")
    plt.ylabel("Predicted Layer")
    plt.title("Predicted vs Actual Layer")
    plt.plot([2,7],[2,7], color='red', linestyle='--')
    scatter_path = os.path.join("outputs", f"pred_vs_actual_{run_id}.png")
    plt.savefig(scatter_path, dpi=300)
    print(f"[INFO] Saved scatter plot to {scatter_path}")
    plt.close()

    # Save model
    best_path = os.path.join("outputs", f"best_model_{run_id}.pt")
    torch.save(model.state_dict(), best_path)
    print(f"[INFO] Saved model to {best_path}")

if __name__ == "__main__":
    main()

