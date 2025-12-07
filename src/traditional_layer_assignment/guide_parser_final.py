import argparse, time
import re, math
import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import defaultdict
from data_struct_util import Node, Net3D, Net2D, CoreGrid3D, Edge, VIA_RESISTANCE
from export_util import write_net_info_to_file, generate_all_route_guides, convert_net2d_to_net3d, convert_net2d_to_net3d_fixed
from rc_tree_util import compute_elmore_delays, assign_layers_with_ndla_elmore, direct_rc_tree, compute_net_overflow, update_grid_usage, update_grid_usage_rip_up, update_grid_usage_2d
import sys

log_file = open("output_log.log", "w")
sys.stdout = log_file
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def layer_to_num(layer_name: str) -> int:
    m = re.search(r'(\d+)$', layer_name)
    if not m:
        raise ValueError(f"Invalid layer name: {layer_name!r}")
    return int(m.group(1))

def parse_route_guide_from_file(file_path):
    route_edges = {}
    fixed_layer_nodes = defaultdict(set)

    with open(file_path, 'r') as f:
        route_text = f.read()
    net_blocks = re.findall(r"([^\s(]+)\s*\(\s*(.*?)\s*\)", route_text, re.DOTALL)

    for net_name, edge_block in net_blocks:
        edges = []
        lines = edge_block.strip().splitlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 6:
                x1, y1, l1, x2, y2, l2 = parts
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)
                l1_num, l2_num = layer_to_num(l1), layer_to_num(l2) #M1 to 1 , M2 to 2...
                edges.append(((x1, y1), (x2, y2), l1_num, l2_num))
                if l1_num == 1:
                    fixed_layer_nodes[net_name].add(((x1, y1), 1))
                if l2_num == 1:
                    fixed_layer_nodes[net_name].add(((x2, y2), 1))
        
        route_edges[net_name] = edges
        for ed in edges:
            (x1, y1), (x2, y2), l1, l2 = ed
    m1_fixed_layer_nodes = dict(fixed_layer_nodes)
    fixed_layer_nodes = defaultdict(list)
    for net, entries in m1_fixed_layer_nodes.items():
        coord_map = defaultdict(set)
        for (coord, layer) in entries:
            coord_map[coord].add(layer)
        for coord, layers in coord_map.items():
            fixed_layer_nodes[net].append((coord, sorted(layers))) #[coord, [L]] FOR EACH NET

    return route_edges, dict(fixed_layer_nodes)

def parse_fixed_nodes_from_def(file_path):
    fixed_nodes = defaultdict(list) #m1 fixed nodes

    with open(file_path, 'r') as f:
        content = f.read()
    die_area_pattern = r"DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)\s*;"
    match = re.search(die_area_pattern, content)
    if match:
        ll = (int(match.group(1)), int(match.group(2)))  # Lower-left
        ur = (int(match.group(3)), int(match.group(4)))  # Upper-right
    else:
        raise ValueError("DIEAREA pattern not found in DEF file.")
    
    die_x = ur[0]-ll[0]
    die_y = ur[1]-ll[1]
    gcell_pattern = r"GCELLGRID\s+[XY]\s+\d+\s+DO\s+(\d+)\s+STEP\s+(\d+)\s*;" 

    matches = re.findall(gcell_pattern, content)

    result_gcell = [(int(do), int(step)) for do, step in matches]

    pin_blocks = re.findall(
        r"-\s+(\S+)\s+\+ NET \1.*?\+ LAYER (?:M|metal)(\d+).*?\+ PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)",
        content, re.DOTALL
    )

    for net_name, layer_str, raw_x, raw_y in pin_blocks:
        x = int(raw_x)
        y = int(raw_y)
        layer_num = layer_to_num(layer_str)

        grid_x = math.floor(x // result_gcell[0][1]) * result_gcell[0][1] + int(result_gcell[0][1]/2)
        grid_y = math.floor(y // result_gcell[1][1]) * result_gcell[1][1] + int(result_gcell[1][1]/2)

        fixed_nodes[net_name].append(((grid_x, grid_y), layer_num))

    
    result = defaultdict(list)
    for net, entries in fixed_nodes.items():
        coord_map = defaultdict(set)
        for (coord, layer) in entries:
            coord_map[coord].add(layer)
        for coord, layers in coord_map.items():
            result[net].append((coord, sorted(layers)))
    return die_x, die_y, result_gcell, dict(result)

def merge_fixed_layer_nodes(*sources):
    merged = defaultdict(list)
    for src in sources:
        for net, nodes in src.items():
            merged[net].extend(nodes)
    return dict(merged)

def populate_nodes_with_edges(net_dict):
    for net in net_dict.values():
        net.organize_edges()
        all_edges_3d = set(net.edges_2d_with_layers).union(net.via_edges)

        all_nodes = set(edge.node1 for edge in all_edges_3d).union(edge.node2 for edge in all_edges_3d)
        for n in all_nodes:
            for edge in all_edges_3d:
                node1 = edge.node1
                node2 = edge.node2
                if n == node1:
                    n.add_edge(node2, edge.layer1, edge.layer2)
                elif n == node2:
                    n.add_edge(node1, edge.layer2, edge.layer1)
    return net_dict
def load_nets_pin_locations(file_path):
    net_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        parts = line.split()
        net_name = parts[0]
        num_pins = int(parts[1])
        i += 1

        nodes = []
        for _ in range(num_pins):
            pin_line = lines[i].strip()
            pin_name, x, y = pin_line.rsplit(maxsplit=2)
            node = Node(x, y, pin_name)
            nodes.append(node)
            i += 1
        net = Net3D(net_name,nodes, num_pins=num_pins)
        net_dict[net_name] = net

    return net_dict


def dump_nets_to_file(net_dict, output_path):
    with open(output_path, 'w') as f:
        for net_name, net in net_dict.items():
            f.write(f"{net_name} {len(net.nodes)}\n")
            for node in net.nodes:
                x, y = node.coord
                f.write(f"{node.pin_name} {x} {y}\n")
            f.write("\n") 


def parse_pin_port_file(file_path, original_net_dict, route_edges, nodes_with_fixed_layers):
    net_dict = {}
    fixed_ports_nets = []
    fixed_port_nets_dict = {}
    skipped_nets_with_no_guides = {}
    i = 0
    with open(file_path, 'r') as f:
        pin_port_text = f.read()
    tech_node_pattern = r"tech platform\s*:\s*(\S+)"
    tech_match = re.search(tech_node_pattern, pin_port_text)

    tech_node = tech_match.group(1) if tech_match else None
    net_blocks = re.findall(r"net name\s*:\s*(.*?)\n(.*?)(?=net name|$)", pin_port_text, re.DOTALL)

    for net_name, block in net_blocks:
        ports_match = re.search(r"num ports\s*:\s*(\d+)", block)
        if not ports_match:
            continue
        num_ports = int(ports_match.group(1))
        if num_ports >= 1:
            fixed_ports_nets.append(net_name)
        total_pins_match = re.search(r"total pins\s*:\s*(\d+)", block)
        if not total_pins_match:
            continue
        total_pins = int(total_pins_match.group(1))

        source_node = None
        sink_nodes = []
        sink_slacks = []
        
        pins = re.findall(r"pin_type\s*:\s*(\w+)\s*with\s*name\s*:\s*(.*?)\s*?\n\s*cap\s*:\s*(-?\d*\.?\d+(?:e[\+\-]?\d+)?),\s*slack_fall\s*:\s*(-?\d*\.?\d+(?:e[\+\-]?\d+)?),\s*slack_rise\s*:\s*(-?\d*\.?\d+(?:e[\+\-]?\d+)?),\s*slack\s*:\s*(-?\d*\.?\d+(?:e[\+\-]?\d+)?)",block)
        ports = re.findall(r"port_type\s*:\s*(\w+)\s*with\s*port\s*name\s*(.*?)\s*?\n\s*cap\s*:\s*(-?\d*\.?\d+(?:e[\+\-]?\d+)?),\s*slack_fall\s*:\s*(-?\d*\.?\d+(?:e[\+\-]?\d+)?),\s*slack_rise\s*:\s*(-?\d*\.?\d+(?:e[\+\-]?\d+)?),\s*slack\s*:\s*(-?\d*\.?\d+(?:e[\+\-]?\d+)?)",block)
        
        fixed = defaultdict(list)
        for coord, layer_info in nodes_with_fixed_layers.get(net_name, []):
            if isinstance(layer_info, list):
                fixed[coord].extend(layer_info)
            else:
                fixed[coord].append(layer_info) 
 
        location_lookup = {}
        if net_name not in original_net_dict.keys(): 
            skipped_nets_with_no_guides[net_name] = net_name
            continue
        net_og = original_net_dict[net_name]
        for node in net_og.nodes:
            location_lookup[node.pin_name] = node.coord  # { "input1/A": (x, y), ... }

        def make_node(x, y, pin_name, cap, role):
            x, y = int(x), int(y)
            cap = float(cap)
            layers = fixed.get((x,y))
            node = Node(x, y, pin_name=pin_name, node_type= role, fixed_layers=layers) 
            node.load_cap = cap
            return node
        
        pin_map = defaultdict(list)
        for pin_type, pin_name, cap, slack_fall, slack_rise, slack in pins:
            pin_name = pin_name.strip()
            slack = float(slack)
            slack_fill = max(-1e6, min(slack, 1e6))
            pin_map[pin_name].append({
                "pin_type": pin_type,
                "cap": float(cap),
                "slack": float(slack_fill)
            })
        port_map = defaultdict(list)
        for port_type,port_name, cap, slack_fall, slack_rise, slack in ports:
            port_name = port_name.strip()
            slack = float(slack)
            slack_fill = max(-1e6, min(slack, 1e6))
            port_map[port_name].append({
                "port_type": port_type,
                "cap": float(cap),
                "slack": float(slack_fill)
            })
        merged_pins = []

        for pin_name, entries in pin_map.items():
            output_entries = [e for e in entries if e['pin_type'] == 'OUTPUT']
            input_entries = [e for e in entries if e['pin_type'] == 'INPUT']
            if len(output_entries) > 1:
                raise ValueError(f"Expected exactly one OUTPUT entry, but found {len(output_entries)}")
            if len(input_entries) > 1:
                raise ValueError(f"Expected exactly one OUTPUT entry, but found {len(input_entries)}")
            if output_entries:
                chosen = output_entries[0]
                pin_type = "OUTPUT"
                cap = chosen["cap"]
                slack = chosen["slack"]
            else:
                pin_type = "INPUT"
                cap = statistics.mean(e["cap"] for e in input_entries)
                slack = statistics.mean(e["slack"] for e in input_entries)
            merged_pins.append((pin_type, pin_name, cap, slack))

        merged_ports = []
        for port_name, entries in port_map.items():
            output_entries = [e for e in entries if e['port_type'] == 'OUTPUT']
            input_entries = [e for e in entries if e['port_type'] == 'INPUT']
            if len(output_entries) > 1:
                raise ValueError(f"Expected exactly one OUTPUT entry, but found {len(output_entries)}")
            if len(input_entries) > 1:
                raise ValueError(f"Expected exactly one INPUT port entry, but found {len(input_entries)}")
            if input_entries:
                chosen = input_entries[0]
                port_type = "INPUT"
                cap = chosen["cap"]
                slack = chosen["slack"]
            else:
                port_type = "OUTPUT"
                cap = statistics.mean(e["cap"] for e in output_entries)
                slack = statistics.mean(e["slack"] for e in output_entries)

            merged_ports.append((port_type, port_name, cap, slack))

        for pin_type, pin_name, cap, slack in merged_pins:
            coord = location_lookup[pin_name]
            if pin_type == "OUTPUT":
                source_node = make_node(coord[0], coord[1],pin_name, cap, "source")
            else:
                sink_nodes.append(make_node(coord[0], coord[1],pin_name, cap, "sink"))
                sink_slacks.append(slack)
        
        for port_type, port_name, cap, slack in merged_ports:
            coord = location_lookup[port_name]
            if port_type == "INPUT":
                source_node = make_node(coord[0], coord[1],port_name, cap, "source")
            else:
                sink_nodes.append(make_node(coord[0], coord[1],port_name, cap, "sink"))
                sink_slacks.append(slack)
                
        if net_name not in route_edges:
            skipped_nets_with_no_guides[net_name] = net_name 
            continue

        edges = route_edges.get(net_name, []) 
        edges_3d = []
        all_src_sink_nodes = []
        all_src_sink_nodes.append(source_node)
        for n in sink_nodes:
            all_src_sink_nodes.append(n)
        node_map = {}  #(x, y) → Node

        def get_or_create_node(x, y, fixed_layers):

            coord = (int(x),int(y))
            if coord not in node_map:
                node_map[coord] = Node(int(x), int(y), fixed_layers=fixed_layers)
            return node_map[coord]
        
        set_vertices = []
        for (x1, y1), (x2, y2), _, _ in edges:
            set_vertices.append((x1,y1))
            set_vertices.append((x2,y2))
        set_vertex = set(set_vertices) 
        for (x1, y1), (x2, y2), l1, l2 in edges:
            layers_1 = fixed.get((x1,y1))
            layers_2 = fixed.get((x2,y2))
            node1 = get_or_create_node(x1, y1, fixed_layers=layers_1)
            node2 = get_or_create_node(x2, y2, fixed_layers=layers_2)
            
            for nod in all_src_sink_nodes:
                if nod.coord == node1.coord:
                    if nod.node_type == "source":
                        node1.pin_name = nod.pin_name
                        node1.load_cap = nod.load_cap
                        node1.node_type = "source"
                    elif nod.node_type == "sink" and node1.node_type != "source":
                        node1.pin_name = nod.pin_name
                        node1.load_cap = nod.load_cap
                        node1.node_type = "sink"

                
                if nod.coord == node2.coord:
                    if nod.node_type == "source":
                        node2.pin_name = nod.pin_name
                        node2.load_cap = nod.load_cap
                        node2.node_type = "source"
                    elif nod.node_type == "sink" and node2.node_type != "source":
                        node2.pin_name = nod.pin_name
                        node2.load_cap = nod.load_cap
                        node2.node_type = "sink"

            e = (node1, node2, l1, l2)
            edges_3d.append(e)
        sinks = []
        src_node = None

        for _, n in node_map.items():
            if n.node_type == 'source':
                src_node = n
            if n.node_type == 'sink':
                sinks.append(n)
        
        if src_node is None:
            raise ValueError(f"No source found for net {net_name}")
        
        net_slack = min(sink_slacks) if sink_slacks else 0.0
        net_dict[net_name] = Net3D(net_name, all_src_sink_nodes, edges_3d=edges_3d,num_pins= total_pins, net_slack=net_slack)
        net_dict[net_name].source = src_node
        net_dict[net_name].sinks = sinks
        i = i+1
    net_dict = populate_nodes_with_edges(net_dict)
    for net_name in fixed_ports_nets:
        if net_name not in net_dict.keys():
            skipped_nets_with_no_guides[net_name] = net_name
            continue
        net_port_3d = net_dict[net_name]
        fixed_port_nets_dict[net_name] = net_port_3d
    return net_dict, fixed_port_nets_dict, tech_node, skipped_nets_with_no_guides

def convert_net3d_to_net2d(net_dict_3d, fixed_ports_net_dict):
    from collections import defaultdict

    net_dict_2d = {}
    net_dict_2d_port_fixed = {}
    net_dict_2d_only_via = {}
    def reset_node_state(node):
        node.edges.clear()
        node.subtree_cap = 0.0
        node.children.clear()
        node.parent_edge = None

    
    for net_name, net3d in fixed_ports_net_dict.items():
        ed_2d_layers = net3d.edges_2d_with_layers
        edges_2d = []
        for ed in ed_2d_layers:
            reset_node_state(ed.node1)
            reset_node_state(ed.node2)

        for ed in ed_2d_layers:
            if ed.edge_type != "via":
                node1, node2 = ed.node1, ed.node2
                layer1, layer2, direction = ed.layer1, ed.layer2, ed.direction
                new_edge = Edge(node1, node2, layer1, layer2, direction)
                node1.add_edge(node2, layer1, layer2)
                node2.add_edge(node1, layer1, layer2)
                if new_edge not in edges_2d:
                    edges_2d.append(new_edge)

        all_nodes = set(edge.node1 for edge in edges_2d).union(edge.node2 for edge in edges_2d)
        source = next((n for n in all_nodes if n.node_type == 'source'), None)
        sinks = [n for n in all_nodes if n.node_type == 'sink']

        net2d = Net2D(
            net_name=net3d.net_name,
            edges_2d=edges_2d,
            source=source,
            sinks=sinks,
            num_pins=net3d.num_pins,
            slack=net3d.slack
        )
        net_dict_2d_port_fixed[net_name] = net2d

    
    for net_name in fixed_ports_net_dict:
        net_dict_3d.pop(net_name, None)
    
    for net3d in net_dict_3d.values():
        if len(net3d.edges_2d_with_layers) == 0:
            net_dict_2d_only_via[net3d.net_name] = net3d

    for net_name in net_dict_2d_only_via:
        net_dict_3d.pop(net_name, None)
    
    
    sorted_nets = sorted(net_dict_3d.values(), key=lambda net: net.slack)
    for net3d in sorted_nets:
        edges_2d = []
        for ed in net3d.edges_2d_with_layers:
            reset_node_state(ed.node1)
            reset_node_state(ed.node2)

        for ed in net3d.edges_2d_with_layers:
            if ed.edge_type != "via":
                node1, node2 = ed.node1, ed.node2
                direction = ed.direction
                new_edge = Edge(node1, node2, layer1=None, layer2=None, direction=direction)
                node1.add_edge(node2, None, None)
                node2.add_edge(node1, None, None)
                if new_edge not in edges_2d:
                    edges_2d.append(new_edge)

        all_nodes = set(edge.node1 for edge in edges_2d).union(edge.node2 for edge in edges_2d)
        source = next((n for n in all_nodes if n.node_type == 'source'), None)
        sinks = [n for n in all_nodes if n.node_type == 'sink']

        net2d = Net2D(
            net_name=net3d.net_name,
            edges_2d=edges_2d,
            source=source,
            sinks=sinks,
            num_pins=net3d.num_pins,
            slack= net3d.slack
        )
        net_dict_2d[net3d.net_name] = net2d

    return net_dict_2d, net_dict_2d_port_fixed, net_dict_2d_only_via

def main():
    start_part1 = time.time()
    parser = argparse.ArgumentParser(description="Parse all net, route, and pin definitions.")
    parser.add_argument('--folder_name', required = True, help = 'Path to folder')
    args = parser.parse_args()

    folder = args.folder_name
    pin_def_file = "5_1_grt.def"
    pin_loc_file = "pin_locations.txt"
    pin_cap_slack_file = "pin_cap_slacks.txt"
    route_guide_file = "segment_route.guide"
    pin_def_file = os.path.join(folder, "5_1_grt.def")
    pin_loc_file = os.path.join(folder, "pin_locations.txt")
    pin_cap_slack_file = os.path.join(folder, "pin_cap_slacks.txt")
    route_guide_file = os.path.join(folder, "segment_route.guide")
    output_guide_file = os.path.join(folder, "timing.guide")
    #start_total = time.time()

    route_edges, fixed_nodes_m1 = parse_route_guide_from_file(route_guide_file)
    die_x, die_y, gcell_step_do_values, fixed_nodes_pinfile = parse_fixed_nodes_from_def(pin_def_file)
    merged_fixed_nodes = merge_fixed_layer_nodes(fixed_nodes_m1, fixed_nodes_pinfile)
    net_dict_3d_without_edges = load_nets_pin_locations(pin_loc_file)
    net_dict_3d, fixed_ports_net_dict, tech_node, skipped_nets_with_no_guides= parse_pin_port_file(pin_cap_slack_file,net_dict_3d_without_edges, route_edges,merged_fixed_nodes)
    #end_part1 = time.time()
    #print(f"Parsing all files took {end_part1 - start_part1:.4f} seconds")

    grid_x = gcell_step_do_values[0][1]
    grid_y = gcell_step_do_values[1][1]
    num_layers =7 
    grid = CoreGrid3D(die_gridx=die_x, die_gridy=die_y, grid_size_x= grid_x, grid_size_y= grid_y, num_layers=num_layers)
    part2 = time.time()
    
    '''
    for net3d in net_dict_3d.values():
        all_edges_3d = set(net3d.edges_2d_with_layers).union(net3d.via_edges)
        all_nodes = set(edge.node1 for edge in all_edges_3d).union(set(edge.node2 for edge in all_edges_3d))
        
        for nod in all_nodes:
            if nod.node_type == 'source':
                src = nod
        delay, _ = compute_elmore_delays(src)
        net3d.delay = delay
    '''

    net_dict_2d_og, _, _ = convert_net3d_to_net2d(net_dict_3d, fixed_ports_net_dict)
    net_dict_2d, net_dict_2d_port_fixed, net_dict_2d_only_via = convert_net3d_to_net2d(net_dict_3d, fixed_ports_net_dict) #ordered based on slack
    
    for net in net_dict_2d.values():
        update_grid_usage_2d(net,grid)
    for net in net_dict_2d_port_fixed.values():
        update_grid_usage_2d(net,grid)
    
    
    for net_name, net in net_dict_2d.items():
        all_nodes = set(edge.node1 for edge in net.edges_2d).union(
                 edge.node2 for edge in net.edges_2d)
        for node in all_nodes:
            if node.node_type == 'source':
                src = node
        direct_rc_tree(src)
    part3 = time.time()
    
    net_dict_3d_reconstructed = {}
    skipped_nets = []
    max_iters = 100
    iters = 0
    i = 0
    #start_part1 = time.time()
    for net_name, net_2d in net_dict_2d_port_fixed.items():
        if net_2d.edges_2d != []:
            update_grid_usage(net_2d, grid)
            net_3d = convert_net2d_to_net3d_fixed(net_2d, fixed_ports_net_dict)
            net_dict_3d_reconstructed[net_2d.net_name] = net_3d
    
    for net_name, net_3d in net_dict_2d_only_via.items():
        net_dict_3d_reconstructed[net_name] = net_3d
    
    for net_name, net_2d in net_dict_2d.items():
        iters = 0
        if net_2d.edges_2d != []:
            net2d_with_layers = assign_layers_with_ndla_elmore(net_2d, grid, num_layers=num_layers, two_pass=True,wD=1.0,wV=0.6,wC = 2)
            if net2d_with_layers is None:
                skipped_nets.append(net_name)
                i += 1
                continue
            update_grid_usage(net2d_with_layers, grid)
            max_overflow_og, total_overflow_og = compute_net_overflow(net_dict_2d_og[net_name], grid)
            max_overflow_r, total_overflow_r = compute_net_overflow(net2d_with_layers, grid)
            while (total_overflow_r > total_overflow_og or max_overflow_r > max_overflow_og/3) and iters  < max_iters +1: #3 because 3 for H and 3 for V
                update_grid_usage_rip_up(net2d_with_layers, grid)
                net2d_with_layers = assign_layers_with_ndla_elmore(net_2d, grid, is_congestion_penalty = True, num_layers=num_layers, two_pass=True, wD=1.0,wV=0.6,wC = 2)
                update_grid_usage(net2d_with_layers,grid)
                max_overflow_r, total_overflow_r = compute_net_overflow(net2d_with_layers, grid)
                iters += 1
            net_3d = convert_net2d_to_net3d(net2d_with_layers)
            net_dict_3d_reconstructed[net_2d.net_name] = net_3d
        i = i+1
    part4 = time.time()
    generate_all_route_guides(net_dict_3d_reconstructed, output_guide_file, tech_node)
    end_total = time.time()
    print(f"DP and assigning layers took {part4 - part3:.6f} seconds")

if __name__ == "__main__":
    main()