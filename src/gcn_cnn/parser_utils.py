import re, math
import statistics
import numpy as np
from collections import defaultdict
from data_struct_util import Node, Edge, Net3D
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
            fixed_layer_nodes[net].append((coord, sorted(layers)))

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
        print("Lower-left corner:", ll)
        print("Upper-right corner:", ur)
    else:
        raise ValueError("DIEAREA pattern not found in DEF file.")
    
    die_x = ur[0]-ll[0]
    die_y = ur[1]-ll[1]
    gcell_pattern = r"GCELLGRID\s+[XY]\s+\d+\s+DO\s+(\d+)\s+STEP\s+(\d+)\s*;"

    matches = re.findall(gcell_pattern, content)

    #Convert to list of (DO, STEP) tuples as integers
    result_gcell = [(int(do), int(step)) for do, step in matches]
    print(f" gcell grid size x  is {result_gcell[0][1]}")
    print(f" gcell grid size y  is {result_gcell[1][1]}")
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
            n.update_via_count_flag
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
            node = Node(x, y, pin_name=pin_name)
            nodes.append(node)
            i += 1
        net = Net3D(net_name,nodes=nodes)
        net_dict[net_name] = net

    return net_dict


def parse_pin_port_file(file_path, original_net_dict, route_edges, nodes_with_fixed_layers):
    net_dict = {}
    fixed_ports_nets = []
    fixed_port_nets_dict = {}
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
            continue
        net_og = original_net_dict[net_name]
        for node in net_og.nodes:
            location_lookup[node.pin_name] = node.coord  # { "input1/A": (x, y), ... }
        def make_node(x, y, slack_rise, slack_fall,cap,pin_name,role, is_pin):
            x, y = int(x), int(y)
            if cap == -1:
                cap = 0.0
            cap = float(cap)
            slack_rise = float(slack_rise)
            slack_fall = float(slack_fall)
            layers = fixed.get((x,y))
            node = Node(x, y, pin_name=pin_name,node_type= role, is_pin=is_pin, fixed_layers=layers)
            node.cap = cap
            node.slack_rise = slack_rise
            node.slack_fall = slack_fall
            return node
        
        pin_map = defaultdict(list)
        for pin_type, pin_name, cap, slack_fall, slack_rise, slack in pins:
            pin_name = pin_name.strip()
            MAX_SLACK = 1E6  

            slack = float(slack)
            slack_rise = float(slack_rise)
            slack_fall = float(slack_fall)
            slack = np.clip(slack, -MAX_SLACK, MAX_SLACK)
            slack_rise = np.clip(slack_rise, -MAX_SLACK, MAX_SLACK)
            slack_fall = np.clip(slack_fall, -MAX_SLACK, MAX_SLACK)
            pin_map[pin_name].append({
                "pin_type": pin_type,
                "cap": float(cap),
                "slack": float(slack),
                "slack_fall": float(slack_fall),
                "slack_rise": float(slack_rise)
            })
        port_map = defaultdict(list)
        for port_type, port_name, cap, slack_fall, slack_rise, slack in ports:
            port_name = port_name.strip()
            MAX_SLACK = 1E6  

            slack = float(slack)
            slack_rise = float(slack_rise)
            slack_fall = float(slack_fall)
            slack = np.clip(slack, -MAX_SLACK, MAX_SLACK)
            slack_rise = np.clip(slack_rise, -MAX_SLACK, MAX_SLACK)
            slack_fall = np.clip(slack_fall, -MAX_SLACK, MAX_SLACK)
            port_map[port_name].append({
                "port_type": port_type,
                "cap": float(cap),
                "slack": float(slack),
                "slack_fall": float(slack_fall),
                "slack_rise": float(slack_rise)
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
                slack_rise= chosen["slack_rise"]
                slack_fall = chosen["slack_fall"]
            else:
                pin_type = "INPUT"
                cap = statistics.mean(e["cap"] for e in input_entries)
                slack = statistics.mean(e["slack"] for e in input_entries)
                slack_fall = statistics.mean(e["slack_fall"] for e in input_entries)
                slack_rise = statistics.mean(e["slack_rise"] for e in input_entries)
            merged_pins.append((pin_type, pin_name, cap, slack, slack_fall, slack_rise))

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
                slack_rise= chosen["slack_rise"]
                slack_fall = chosen["slack_fall"]
            else:
                port_type = "OUTPUT"
                cap = statistics.mean(e["cap"] for e in output_entries)
                slack = statistics.mean(e["slack"] for e in output_entries)
                slack_fall = statistics.mean(e["slack_fall"] for e in output_entries)
                slack_rise = statistics.mean(e["slack_rise"] for e in output_entries)
            merged_ports.append((port_type, port_name, cap, slack, slack_fall, slack_rise))

        for pin_type, pin_name, cap, slack, slack_fall, slack_rise in merged_pins:
            coord = location_lookup[pin_name]
            if pin_type == "OUTPUT":
                source_node = make_node(coord[0],coord[1],pin_name=pin_name, cap =cap, slack_fall=slack_fall, slack_rise = slack_rise, role="source", is_pin = True)
            else:
                sink_nodes.append(make_node(coord[0],coord[1], pin_name=pin_name,cap=cap, slack_fall = slack_fall,slack_rise = slack_rise, role= "sink", is_pin = True))
                sink_slacks.append(slack)
        for port_type, port_name, cap, slack, slack_fall, slack_rise in merged_ports:
            coord = location_lookup[port_name]
            if port_type == "INPUT":
                source_node = make_node(coord[0],coord[1],pin_name=port_name, cap =cap, slack_fall=slack_fall, slack_rise = slack_rise, role="source", is_pin = True)
            else:
                sink_nodes.append(make_node(coord[0],coord[1],pin_name=port_name, cap=cap, slack_fall = slack_fall,slack_rise = slack_rise, role= "sink", is_pin = True))
                sink_slacks.append(slack)
        if net_name not in route_edges:
            continue

        edges = route_edges.get(net_name, [])
        edges_3d = []
        nodes = {}
        all_src_sink_nodes = []
        all_src_sink_nodes.append(source_node)

        for n in sink_nodes:
            all_src_sink_nodes.append(n)
        node_map = {}  #(x, y) → Node
        net_slack = min(sink_slacks) if sink_slacks else 0.0
        def get_or_create_node(x, y, fixed_layers):
            coord = (int(x),int(y))
            if coord not in node_map:
                node_map[coord] = Node(int(x), int(y), fixed_layers=fixed_layers)
            return node_map[coord]
        
        set_vertices = []
        total_net_length = 0
        for (x1, y1), (x2, y2), l1, l2 in edges:
            set_vertices.append((x1,y1))
            set_vertices.append((x2,y2))
            total_net_length += math.sqrt((x1-x2)**2 +(y1-y2)**2)
        set_vertex = set(set_vertices)
        
        
        for (x1, y1), (x2, y2), l1, l2 in edges:
            layers_1 = fixed.get((x1,y1))
            layers_2 = fixed.get((x2,y2))
            node1 = get_or_create_node(x1, y1, fixed_layers=layers_1)
            node2 = get_or_create_node(x2, y2, fixed_layers=layers_2)
            
            for nod in all_src_sink_nodes:
                if nod.coord == node1.coord:
                    if nod.node_type == "source":
                        node1.cap = nod.cap
                        node1.pin_name = nod.pin_name
                        node1.slack_rise = nod.slack_rise
                        node1.slack_fall = nod.slack_fall
                        node1.is_pin = nod.is_pin
                        node1.node_type = "source"

                    elif nod.node_type =="sink" and node1.node_type != "source":
                        node1.cap = nod.cap
                        node1.pin_name = nod.pin_name
                        node1.slack_rise = nod.slack_rise
                        node1.slack_fall = nod.slack_fall
                        node1.is_pin = nod.is_pin
                        node1.node_type = "sink"
                if nod.coord == node2.coord:
                    if nod.node_type == "source":
                        node2.cap = nod.cap
                        node2.pin_name = nod.pin_name
                        node2.slack_rise = nod.slack_rise
                        node2.slack_fall = nod.slack_fall
                        node2.is_pin = nod.is_pin
                        node2.node_type = "source"

                    elif nod.node_type == "sink" and node2.node_type != "source":
                        node2.cap = nod.cap
                        node2.pin_name = nod.pin_name
                        node2.slack_rise = nod.slack_rise
                        node2.slack_fall = nod.slack_fall
                        node2.is_pin = nod.is_pin
                        node2.node_type = "sink"
            
            nodes[node1.name] = node1
            nodes[node2.name] = node2
            
            e = Edge(node1, node2, l1, l2)
            e.net_slack = net_slack
            e.net_length = total_net_length
            e.num_pins = total_pins
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
        
        net_dict[net_name] = Net3D(net_name, edges_3d=edges_3d, nodes=nodes)
        i= i+1
    net_dict = populate_nodes_with_edges(net_dict)
    for net_name in fixed_ports_nets:
        if net_name not in net_dict.keys():
            continue
        net_port_3d = net_dict[net_name]
        fixed_port_nets_dict[net_name] = net_port_3d
    return net_dict, fixed_port_nets_dict, tech_node

