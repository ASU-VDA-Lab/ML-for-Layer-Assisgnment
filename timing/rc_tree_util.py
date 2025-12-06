from data_struct_util import Net2D, Edge
import math
from export_util import RobustNormalizer
from collections import defaultdict, deque
from data_struct_util import Net3D, layer_configs, VIA_RESISTANCE
from itertools import product


def direct_rc_tree(source): 
    visited = set()
    def dfs(node):
        visited.add(node)
        children = []
        for edge in node.get_flat_edges():
            other = edge.node2 if edge.node1 == node else edge.node1
            if other not in visited:
                R = edge.resistance if hasattr(edge, 'resistance') else 0.0
                C = edge.capacitance if hasattr(edge, 'capacitance') else 0.0
                other.parent_edge = edge
                dfs(other)
                children.append((other, R, C))
                
        node.children = children

    dfs(source)


def compute_elmore_delays(source):
    direct_rc_tree(source)
    
    def compute_subtree_caps(node):
        load = max(node.load_cap, 0.0)
        total = load
        
        for child, _, C in node.children:
            C = max(C, 0.0)
            total += C + compute_subtree_caps(child)
        node.subtree_cap = total
        return total

    total_cap = compute_subtree_caps(source)
    
    delays = {}
    def dfs_delay(node, delay_so_far, r_from_parent, edge_cap_from_parent):
        if r_from_parent is not None :
            
            via_res = len(node.get_via_edges())*VIA_RESISTANCE
            delay_here = delay_so_far + (r_from_parent +via_res)* (node.subtree_cap + edge_cap_from_parent)
        else:
            via_res = len(node.get_via_edges())*VIA_RESISTANCE
            delay_here = 0.0 + via_res * node.subtree_cap
        if node.load_cap > 0:
            delays[node.name]= delay_here 
        
        for child, R, C in node.children:
            dfs_delay(child, delay_here, R, C)
            
        
    dfs_delay(source, 0.0, None, 0.0)
    avg_delay = sum(delays.values())/len(delays.values()) if delays else 0
    return avg_delay, total_cap



def get_subtree_delay_from_node(node):
    avg_delay, total_cap = compute_elmore_delays(node)
    return avg_delay, total_cap

def legal_layers_for_edge_direction(direction, num_layers=7):
    if direction in ('H', 'HORIZONTAL'):
        return [layer for layer, config in layer_configs.items() if config[0] == 'H' and layer != 1 and layer <= num_layers]
    elif direction in ('V', 'VERTICAL'):
        return [layer for layer, config in layer_configs.items() if config[0] == 'V' and layer != 1 and layer <= num_layers]
    return [layer for layer in layer_configs.keys() if layer != 1 and layer <= num_layers]


def assign_layers_with_ndla_elmore(net2d, grid=None, is_congestion_penalty=False,num_layers=7,
                                   two_pass=True,wD=1.0,wV=0.6,wC=0.6,clamp_k=3.0,cong_power = 2.0):
    src = None
    all_nodes = set(ed.node1 for ed in net2d.edges_2d).union(set(ed.node2 for ed in net2d.edges_2d))
    for n in all_nodes:
        if n.node_type == 'source':
            src = n
            break
    if src is None:
        raise ValueError(f"No source found for net {net2d.net_name}")

    direct_rc_tree(src)
    normalizer = RobustNormalizer(clamp_k=clamp_k)

    def run_dp(normalize=False):
        dp = defaultdict(dict)
        def bottom_up_dp(node, dp_table):
            for child, _, _ in node.children:
                bottom_up_dp(child, dp_table)

            if node.parent_edge:
                node1 = node.parent_edge.node1
                node2 = node.parent_edge.node2
                direction = 'H' if node1.coord[1] == node2.coord[1] else 'V'
                legal_layers = legal_layers_for_edge_direction(direction, num_layers)
                for l in legal_layers:
                    if l in (8,9):
                        raise ValueError("hey i am wrong please chcek")
                fixed = node.fixed_layers if getattr(node, 'fixed_layers', None) else []
                if not legal_layers:
                    (f"[WARNING] No legal layers for node {node.name}")
                    return

                for layer in legal_layers:
                    R = layer_configs[layer][2] * node.parent_edge.length
                    C = layer_configs[layer][3] * node.parent_edge.length
                    load = max(getattr(node, 'load_cap', 0.0), 0.0)
                    delay = 0
                    layer_set = [layer] +fixed
                    child_choices = []
                    via_count = 0
                    for child, _, _ in node.children:
                        best_layer = min(dp_table[child], key=lambda l: dp_table[child][l]["cost"])
                        child_choices.append((child, best_layer))
                        layer_set.append(best_layer)
                        subtree_cap = getattr(child, 'subtree_cap', 0.0)
                        branch_vias = abs(layer - best_layer)
                        via_res = branch_vias * VIA_RESISTANCE
                        delay += (R + via_res) * (subtree_cap + C + load) + dp_table[child][best_layer]["delay"]
                        via_count += dp_table[child][best_layer]["via_count"] + branch_vias
                   

                    if not node.children:
                        flist_leaf = node.fixed_layers
                        if flist_leaf:
                            via_count += abs(max(flist_leaf) - layer)
                        via_res = via_count * VIA_RESISTANCE
                        delay = (R +via_res)* (C + load)

                    L = node.parent_edge.length
                    phi = 0.0
                    if is_congestion_penalty and grid:
                        idx = grid.layer_index_map.get(layer)
                        cells = grid.get_grid_cells_crossed(node1.coord[0], node1.coord[1],node2.coord[0], node2.coord[1])
                        for (x, y) in cells:
                            capacity = grid.capacity[x][y][idx]
                            usage = grid.usage[x][y][idx]
                            if capacity > 0:
                                u = usage / capacity
                                if u > 1.0:
                                    phi += (u - 1.0) ** cong_power
                    if not normalize:
                        # pass 1: just collect stats
                        normalizer.observe(delay, via_count, phi)
                        total_cost = delay  # placeholder
                    else:
                        # pass 2: normalized blend
                        D_hat = normalizer.norm_D(delay)
                        V_hat = normalizer.norm_V(via_count)
                        C_hat = normalizer.norm_C(phi if is_congestion_penalty else 0.0)
                        total_cost = wD * D_hat + wV * V_hat + wC * C_hat
             

                    dp_table[node][layer] = {
                        "delay": delay,
                        "via_count": via_count,
                        "cong": phi,
                        "cost": total_cost,
                        "child_choices": child_choices
                    }

            else:
                flist = node.fixed_layers
                lf_root = max(flist) if flist else None
                default_layers = [l for l in (grid.layer_names if grid else sorted(layer_configs)) if l != 1 and l <= num_layers]
                for layer in default_layers:
                    via_count = 0
                    delay = 0.0
                    if lf_root is not None and lf_root != layer:
                        initial_stack = abs(lf_root - layer)
                        via_count += initial_stack
                        delay += (initial_stack * VIA_RESISTANCE) * getattr(node, "subtree_cap", 0.0)
                    if not normalize:
                        normalizer.observe(delay, via_count, 0.0)
                        cost = delay
                    else:
                        D_hat = normalizer.norm_D(delay)
                        V_hat = normalizer.norm_V(via_count)
                        C_hat = normalizer.norm_C(0.0)
                        cost = wD * D_hat + wV * V_hat + wC * C_hat
                    dp_table[node][layer] = {
                        "delay": delay,
                        "via_count": via_count,
                        "cong": 0.0,
                        "cost": cost,
                        "child_choices": [
                            (child, min(dp_table[child], key=lambda l: dp_table[child][l]["cost"]))
                            for child, _, _ in node.children
                        ]
                    }

        def top_down_assign_layers(node, dp_table, chosen_layer=None):
            if node.parent_edge and chosen_layer is not None:
                node.parent_edge.layer1 = chosen_layer
                node.parent_edge.layer2 = chosen_layer

            if chosen_layer is None:
                chosen_layer = min(dp_table[node], key=lambda l: dp_table[node][l]["cost"])
            for child, child_layer in dp_table[node][chosen_layer]["child_choices"]:
                top_down_assign_layers(child, dp_table, child_layer)

        bottom_up_dp(src, dp)
        top_down_assign_layers(src,dp)
        return dp
    
    if two_pass:
        _ = run_dp(normalize=False)   #pass 1: collect stats only
        normalizer.fit()              #compute medians / scales
        dp_final = run_dp(normalize=True)  #pass 2: real costs
    else:
        dp_final = run_dp(normalize=False)
    net2d.edges_2d.clear()
    visited_nodes = set()
    def dfs_parent_edge(node):
        visited_nodes.add(node)
        
        for child, _, _ in node.children:
            if child not in visited_nodes:
                dfs_parent_edge(child)
        parent_edge = node.parent_edge
        if parent_edge is not None:
            net2d.edges_2d.append(parent_edge)
    dfs_parent_edge(src)
    return net2d


def update_grid_usage(net, core_grid):
    for ed in net.edges_2d:
        layer = ed.layer1
        idx = core_grid.layer_index_map.get(layer)
        (x1, y1) = ed.node1.coord
        (x2, y2) = ed.node2.coord
        cells = core_grid.get_grid_cells_crossed(x1, y1, x2, y2)
        for r, c in cells:
            core_grid.usage[r][c][idx] += 1

def update_grid_usage_2d(net, core_grid):
    for ed in net.edges_2d:
        (x1, y1) = ed.node1.coord
        (x2, y2) = ed.node2.coord
        cells = core_grid.get_grid_cells_crossed(x1, y1, x2, y2)
        for r, c in cells:
            core_grid.usage_2d[r][c] += 1
    
def compute_net_overflow(net, core_grid):
    edges = net.edges_2d

    net_total_overflow = 0
    net_max_overflow = 0

    for ed in edges:
        layer = ed.layer1
        idx = core_grid.layer_index_map.get(layer)

        (x1, y1) = ed.node1.coord
        (x2, y2) = ed.node2.coord
        cells = core_grid.get_grid_cells_crossed(x1, y1, x2, y2)
        edge_max = 0
        edge_total = 0

        if idx is None:
            direc = "V" if x1 == x2 else "H"
            candidate_layer_ids = [lid for lid, cfg in layer_configs.items() if cfg[0] == direc and lid != 1]
            for r,c in cells:
                use_og = core_grid.usage_2d[r][c]
                ovf = 0
                for lid in candidate_layer_ids:
                    i = core_grid.layer_index_map.get(lid)
                    cap_og  = core_grid.capacity[r][c][i]
                    ovf_og = max(use_og - cap_og, 0)
                    ovf = max(ovf, ovf_og)
                edge_total += ovf
                edge_max = max(edge_max, ovf)
                
        else:
            for r, c in cells:
                use = core_grid.usage[r][c][idx]
                cap = core_grid.capacity[r][c][idx]
                ovf = max(use - cap, 0)
                edge_total += ovf
                edge_max = max(edge_max, ovf)
                
            
        net_total_overflow += edge_total
        net_max_overflow = max(net_max_overflow, edge_max)

    return net_max_overflow, net_total_overflow


def update_grid_usage_rip_up(net, core_grid):
    edges_2d_with_layers = net.edges_2d
    for ed in edges_2d_with_layers:
        layer = ed.layer1
        idx = core_grid.layer_index_map.get(layer)
        (x1, y1) = ed.node1.coord
        (x2, y2) = ed.node2.coord
        cells = core_grid.get_grid_cells_crossed(x1, y1, x2, y2)
        for r, c in cells:
            core_grid.usage[r][c][idx] -= 1
