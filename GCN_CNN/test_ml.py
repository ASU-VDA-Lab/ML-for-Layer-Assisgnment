
import argparse
import os
import re
import glob as globmod
from typing import Optional, Tuple, List
from collections import Counter

import torch
from sklearn.metrics import classification_report, f1_score

from gcn_cnn_util import EdgeLayerPredictor
from data_struct_util import layer_configs, CoreGrid3D, Net3D, Edge
from guide_parser_with_latest import build_data_from_shared_nets, load_norm_stats, normalize_graph_inplace, class_to_layer, orient_of_layer, class_mask_from_orient, build_net_cnn_feats,  RCLogitBiasUrgent
from parser_utils import (
    parse_route_guide_from_file,
    parse_fixed_nodes_from_def,
    merge_fixed_layer_nodes,
    load_nets_pin_locations,
    parse_pin_port_file,
)
from helpers import design_of




#Guide reconstruction
metal_naming = {
    "asap7": lambda l: f"M{l}",
    "nangate45": lambda l: f"metal{l}",
}


def generate_all_route_guides(net3d_dict_reconstructed, output_file, tech_node):
    if tech_node is None:
        raise ValueError("tech node not provided")
    
    layer_str = metal_naming[tech_node]
    with open(output_file, 'w') as f:
        for net_name, net3d in net3d_dict_reconstructed.items():
            if net3d.edges_3d == []:
                continue
            f.write(f'{net_name}\n')
            f.write('(\n')
            for edge in net3d.edges_3d:
                n1, n2, l1, l2 = edge.node1, edge.node2, edge.layer1, edge.layer2
                x1, y1 = n1.coord
                x2, y2 = n2.coord
                if n1.coord == n2.coord:
                    f.write(f'{x1} {y1} {layer_str(l1)} {x1} {y1} {layer_str(l2)}\n')
                else:
                    f.write(f'{min(x1,x2)} {min(y1,y2)} {layer_str(l1)} {max(x1,x2)} {max(y1,y2)} {layer_str(l2)}\n')
            f.write(')\n')


def convert_net3d_to_for_reconstruction(net, net_dict, fixed: bool = False):
    edges_3d_l = set()
    visited_l = set()

    via_src = net_dict.get(net.net_name, None)
    via_edges_3d = getattr(via_src, 'via_edges', []) if fixed and via_src is not None else []

    all_nodes_l = set(e.node1 for e in net.edges_2d_with_layers).union(
        set(e.node2 for e in net.edges_2d_with_layers)
    )
    for n_l in all_nodes_l:
        n_l.edges.clear()

    for e in via_edges_3d:
        e.node1.add_edge(e.node2, e.layer1, e.layer2)
        e.node2.add_edge(e.node1, e.layer1, e.layer2)
        via_e_key = (e.node1, e.node2, e.layer1, e.layer2)
        via_ed = Edge(e.node1, e.node2, e.layer1, e.layer2)
        if via_e_key not in visited_l:
            visited_l.add(via_e_key)
            edges_3d_l.add(via_ed)

    #Add horizontal/vertical edges based on layer assignment
    for edge_l in net.edges_2d_with_layers:
        if fixed:
            layer_assigned = getattr(edge_l, 'actual_layer', None)
        else:
            layer_assigned = getattr(edge_l, 'predicted_layer', None)
            
        if layer_assigned is None:
            continue
        edge_l.node1.add_edge(edge_l.node2, layer_assigned, layer_assigned)
        edge_l.node2.add_edge(edge_l.node1, layer_assigned, layer_assigned)
        e_key_l = (edge_l.node1, edge_l.node2, layer_assigned, layer_assigned)
        ed_key_l = Edge(edge_l.node1, edge_l.node2, layer_assigned, layer_assigned)
        if e_key_l not in visited_l:
            visited_l.add(e_key_l)
            edges_3d_l.add(ed_key_l)

    def add_vias_between(nod, lo, hi):
        for l in range(lo, hi):
            via_key_l = (nod, nod, l, l + 1)
            via_edge_l = Edge(nod, nod, l, l + 1)
            nod.add_edge(nod, l, l + 1)
            if via_key_l not in visited_l:
                visited_l.add(via_key_l)
                edges_3d_l.add(via_edge_l)

    for node_l in all_nodes_l:
        predicted_layers = [getattr(e, 'predicted_layer', None) for e in node_l.edges]
        actual_layers = [getattr(e, 'actual_layer', None) for e in node_l.edges]
        if fixed:
            connected_layers = [l for l in actual_layers if l is not None]
        else:
            connected_layers = [l for l in predicted_layers if l is not None]
            
        if not connected_layers:
            continue
        lo_l, hi_l = min(connected_layers), max(connected_layers)
        if hi_l != lo_l:
            add_vias_between(node_l, lo_l, hi_l)
        fixed_layers = getattr(node_l, 'fixed_layers', None) or [1]
        for fixed_l in fixed_layers:
            for lay in connected_layers:
                if fixed_l != lay:
                    lo, hi = min(fixed_l, lay), max(fixed_l, lay)
                    add_vias_between(node_l, lo, hi)

    all_nodes_re = set(ed.node1 for ed in net.edges_2d_with_layers).union(
        set(ed.node2 for ed in net.edges_2d_with_layers)
    )
    net3d_l = Net3D(net_name=net.net_name, nodes=all_nodes_re, edges_3d=list(edges_3d_l))
    net3d_l.organize_edges()
    return net3d_l


def reconstruct_and_write_guides(net_dict_3d, fixed_ports_net_dict, out_path, tech_node):
    net_dict_3d_reconstructed = {}
   
    for net_name, net in net_dict_3d.items():
        net_re = convert_net3d_to_for_reconstruction(net, net_dict_3d, fixed=False)
        net_dict_3d_reconstructed[net_name] = net_re

    for net_name, net in fixed_ports_net_dict.items():
        net_re = convert_net3d_to_for_reconstruction(net, fixed_ports_net_dict, fixed=True)
        net_dict_3d_reconstructed[net_name] = net_re
    generate_all_route_guides(net_dict_3d_reconstructed, out_path, tech_node)


#Data loader

def load_datapoint(folder):
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

    for net_name in fixed_ports_net_dict:
        net_dict_3d.pop(net_name,None)
        
    graph_data, edge_samples = build_data_from_shared_nets(net_dict_3d, grid.congestion_image)
    graph_data.congestion_img = grid.congestion_image.to(dtype=torch.float32)
    
    L, H, W = grid.congestion_image.shape  # [6, H, W]
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
        print(f"datapoint has 0 edge samples: {folder}", flush=True)
        raise ValueError(f"check {folder} as it doesnt have any edge sample")

    print(f"[LOAD]   nodes={graph_data.x.size(0)} edges(attr)={graph_data.edge_attr.size(0)} samples={len(edge_samples)}",
              flush=True)
    return graph_data, edge_samples, net_dict_3d, fixed_ports_net_dict, tech_node, (die_x, die_y, grid_x, grid_y)

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

LOSS_TAG_MAP = {
    "wfl": "wfl", "weighted_focal": "wfl",
    "fl": "fl", "focal": "fl",
    "wce": "wce", "weighted_ce": "wce",
}

RUN_RE = re.compile(
    r"best_model_(?P<scenario>opt[123])_test-(?P<testtag>[^_]*)_e(?P<epochs>\d+)_(?P<loss>wce|fl|wfl)\.pt$"
)

def design_of_path(p: str) -> str:
    return os.path.basename(os.path.dirname(os.path.normpath(p)))

def infer_test_tag_from_folders(folders: List[str]) -> str:
    designs = sorted({design_of_path(f) for f in folders})
    return "_".join(designs) if designs else "none"

def pick_latest(paths: List[str]) -> Optional[str]:
    return max(paths, key=os.path.getmtime) if paths else None

import os, re
import glob as globmod

def find_run_files(outputs_dir: str, scenario: str, test_tag: str, prefer_loss: str | None):
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    loss_order = []
    if prefer_loss in ("wfl", "fl", "wce"):
        loss_order.append(prefer_loss)
    for lt in ("wfl", "fl", "wce"):
        if lt not in loss_order:
            loss_order.append(lt)

    for loss_tag in loss_order:
        pattern = os.path.join(
            outputs_dir,
            f"best_model_{scenario}_test-*{test_tag}*_e*_{loss_tag}.pt"
        )
        hits = globmod.glob(pattern)
        if hits:
            ckpt = max(hits, key=os.path.getmtime)
            m = re.search(r"best_model_(.+)\.pt$", os.path.basename(ckpt))
            run_id = m.group(1) if m else None
            norm = os.path.join(outputs_dir, f"norm_{run_id}.pt") if run_id else None
            if norm and os.path.exists(norm):
                return ckpt, norm, scenario

    all_best = sorted(os.path.basename(p) for p in globmod.glob(os.path.join(outputs_dir, "best_model_*.pt")))
    nearby = [p for p in all_best if f"_{scenario}_" in p]
    raise FileNotFoundError(
        "No matching checkpoints found.\n"
        f"- looked for: scenario={scenario}, test_tag~='{test_tag}', prefer_loss={prefer_loss}\n"
        f"- outputs dir: {outputs_dir}\n"
        f"- available best_model files (showing up to 20):\n  " + "\n  ".join(all_best[:20]) + "\n"
        f"- with scenario '{scenario}' (up to 20):\n  " + "\n  ".join(nearby[:20])
    )




@torch.no_grad()
def evaluate_and_assign_from_folders(model, folders, device, stats, guide_name="ml_pred.guide"):
    model.eval()
    total_correct, total_samples = 0, 0
    all_preds, all_labels = [], []

    for folder in folders:
        print(f"[LOAD] folder: {folder}")
        out_folder = folder

        data, samples, net_dict_3d, fixed_ports_net_dict, tech_node, _ = load_datapoint(folder)
        norm_device = data.x.device if getattr(data, "x", None) is not None else torch.device("cpu")
        data, samples = normalize_graph_inplace(data, samples, stats, device=norm_device)
        
        gcn_out = model.gcn_encoder(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        net_vec_table = build_net_cnn_feats(model, data, device)
        name_to_idx = {n: i for i, n in enumerate(data.net_names)}
        net_idx_per_sample = torch.tensor([name_to_idx[s["net_key"]] for s in samples],dtype=torch.long, device=device)


        
        

        
        def build_edge_lookup(net_dict):
            lookup = {}
            for net_name, net in net_dict.items():
                for edge in net.edges_2d_with_layers:
                    key = (net_name, tuple(sorted([edge.node1.name, edge.node2.name])))
                    lookup[key] = edge
            return lookup
        edge_lookup = build_edge_lookup(net_dict_3d)
        fixed_lookup = build_edge_lookup(fixed_ports_net_dict)

        
        src = torch.tensor([s["src"] for s in samples], dtype=torch.long, device=device)
        dst = torch.tensor([s["dst"] for s in samples], dtype=torch.long, device=device)
        eattr = torch.stack([s["edge_attr"] for s in samples]).to(device)
        labels_all = torch.tensor([s["label"] for s in samples], dtype=torch.long, device=device)
        orients = [s.get("orient", None) for s in samples]
        net_keys = [s["net_key"] for s in samples]
        src_names = [s["src_name"] for s in samples]
        dst_names = [s["dst_name"] for s in samples]

        N = labels_all.numel()
        bs = _auto_edge_batch_size(device, N)
        start = 0
        while start < N:
            end = min(start + bs, N)
            sl = slice(start, end)

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

            preds = out.argmax(1)
            labels = labels_all[sl]

            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()

            all_labels.extend((labels + 2).tolist())

            for i, pred in enumerate(preds.tolist()):
                predicted_layer = pred+2
                key = (net_keys[start + i], tuple(sorted([src_names[start + i], dst_names[start + i]])))
                if key in fixed_lookup:
                    continue
                if key in edge_lookup:
                    
                    edge_lookup[key].predicted_layer = predicted_layer
                    if edge_lookup[key].actual_layer != edge_lookup[key].predicted_layer:
                        print("here pred layer is different from actual layer")
                all_preds.append(predicted_layer)

            start = end

        #Now write the guide with predictions
        #out_path = os.path.join(out_folder, guide_name)
        #reconstruct_and_write_guides(net_dict_3d, fixed_ports_net_dict, out_path, tech_node)
        #print(f"[GUIDE] wrote: {out_path}")

    acc = (total_correct / total_samples) if total_samples else float('nan')
    return acc, all_labels, all_preds



def main():
    p = argparse.ArgumentParser(description="Standalone tester for EdgeLayerPredictor + guide writer")

    p.add_argument("--prefer-loss", default=None,
                   help="Loss tag to prefer when auto-picking ckpt: wce|fl|wfl (or full names).")
    p.add_argument("--scenario", choices=["opt1", "opt2", "opt3"], default=None,
                   help="Force selecting a specific scenario; affects auto-pick and guide name.")
    p.add_argument("--test-designs-tag", default=None,
                   help="Override test tag used in training run_id (e.g., 'aes' or 'aes_jpeg').")

    p.add_argument("--ckpt", default=None, help="Path to model .pt (state_dict). If omitted, auto-find in outputs/")
    p.add_argument("--norm-path", default=None, help="Path to normalization stats .pt. If omitted, auto-find in outputs/")

    p.add_argument("--folders", nargs='+', required=True, help="One or more datapoint folders")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--guide-name", default=None, help="Filename for generated guide; defaults to ml_pred_optX.guide")

    args = p.parse_args()
    device = torch.device(args.device)


    ckpt = args.ckpt
    norm = args.norm_path
    scenario_from_name = args.scenario 

    if ckpt is None or norm is None:
        test_tag = args.test_designs_tag or infer_test_tag_from_folders(args.folders)
        outputs_dir = "outputs"
        ckpt, norm, parsed_scenario = find_run_files(
            outputs_dir=outputs_dir,
            test_tag=test_tag,
            prefer_loss=args.prefer_loss,
            scenario=args.scenario
        )
        scenario_from_name = args.scenario or parsed_scenario
        print(f"[AUTO] Using ckpt: {ckpt}")
        print(f"[AUTO] Using norm : {norm}")
        print(f"[AUTO] Scenario   : {scenario_from_name}")

    node_feat_dim = 5
    edge_feat_dim = 13
    cnn_dim = 64
    gcn_dim = 64
    hidden_dim = 128
    num_classes = 6

    model = EdgeLayerPredictor(node_feat_dim=node_feat_dim,
                               edge_feat_dim=edge_feat_dim,
                               cnn_dim=cnn_dim,
                               gcn_dim=gcn_dim,
                               hidden_dim=hidden_dim,
                               num_classes=num_classes).to(device)
    
    model.rc_bias_mod = RCLogitBiasUrgent(slack_idx=0, r_start=1, c_start=7).to(device)
    model.bias_gain = torch.nn.Parameter(torch.tensor(0.0, device=device))

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    stats = load_norm_stats(norm, map_location="cpu")
    
    guide_name = args.guide_name or (f"ml_pred_{scenario_from_name}.guide" if scenario_from_name else "ml_pred.guide")
    print(f"[GUIDE] Output name: {guide_name}")
    
    acc, labels, preds = evaluate_and_assign_from_folders(
        model, args.folders, device, stats
    )

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(labels, preds, labels=[2,3,4,5,6,7], digits=4))
    print("Macro F1:", f1_score(labels, preds, labels=[2,3,4,5,6,7], average="macro"))


if __name__ == "__main__":
    main()
