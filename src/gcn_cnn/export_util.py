from data_struct_util import Net3D, Edge
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
            f.write(f'{net_name}\n')
            f.write('(\n')
            for edge in net3d.edges_3d:
                n1, n2, l1, l2 = edge
                x1, y1 = n1.coord
                x2, y2 = n2.coord
                if n1.coord == n2.coord:
                    f.write(f'{x1} {y1} {layer_str(l1)} {x1} {y1} {layer_str(l2)}\n')
                else:
                    f.write(f'{min(x1,x2)} {min(y1,y2)} {layer_str(l1)} {max(x1,x2)} {max(y1,y2)} {layer_str(l2)}\n')
            f.write(')\n')

def convert_net2d_to_net3d(net, net_dict, fixed=False):
    edges_3d_l = set()
    visited_l = set()
    via_edges_3d = net_dict[net.net_name].via_edges if fixed== True else []
    all_nodes_l = set(edge_l.node1 for edge_l in net.edges_2d_with_layers).union(set(edge_l.node2 for edge_l in net.edges_2d_with_layers))
    
    for n_l in all_nodes_l:
        n_l.edges.clear()
    for e in via_edges_3d:
        e.node1.add_edge(e.node2, e.layer1, e.layer2)
        e.node2.add_edge(e.node1, e.layer1, e.layer2)
        via_e_key =(e.node1, e.node2, e.layer1, e.layer2)
        via_ed = Edge(e.node1, e.node2, e.layer1, e.layer2)
        if via_e_key not in visited_l:
            visited_l.add(via_e_key)
            edges_3d_l.add(via_ed)
            
    for edge_l in net.edges_2d_with_layers:
        layer_assigned = edge_l.predicted_layer if fixed == False else edge_l.actual_layer
        edge_l.node1.add_edge(edge_l.node2, layer_assigned, layer_assigned)
        edge_l.node2.add_edge(edge_l.node1, layer_assigned, layer_assigned)
        e_key_l = (edge_l.node1, edge_l.node2, layer_assigned, layer_assigned)
        ed_key_l = Edge(edge_l.node1, edge_l.node2, layer_assigned, layer_assigned)
        if e_key_l not in visited_l:
            visited_l.add(e_key_l)
            edges_3d_l.add(ed_key_l)

    def add_vias_between(nod, lo, hi):
        """Adds via edges between layers lo to hi for a single node."""
        for l in range(lo, hi):
            via_key_l = (nod, nod, l, l + 1)
            via_edge_l = Edge(nod, nod, l, l+1)
            nod.add_edge(nod, l, l+1)
            if via_key_l not in visited_l:
                visited_l.add(via_key_l)
                edges_3d_l.add(via_edge_l)

    for node_l in all_nodes_l:
        predicted_layers = [e.predicted_layer for e in node_l.edges if e.predicted_layer is not None]
        actual_layers = [e.actual_layer for e in node_l.edges if e.actual_layer is not None]
        connected_layers = predicted_layers if fixed == False else actual_layers
        if not connected_layers:
            continue

        min_l, max_l = min(connected_layers), max(connected_layers)
        if max_l != min_l:
            add_vias_between(node_l, min_l, max_l)
        fixed_layers = node_l.fixed_layers if node_l.fixed_layers else [1]

        for fixed_l in fixed_layers:
            for lay in connected_layers:
                if fixed_l != lay:
                    lo, hi = min(fixed_l, lay), max(fixed_l, lay)
                    add_vias_between(node_l, lo, hi)   

    all_nodes_re = set(ed.node1 for ed in net.edges_2d_with_layers).union(set(ed.node2 for ed in net.edges_2d_with_layers))
    
    
    net3d_l = Net3D(
        net_name=net.net_name,
        nodes=all_nodes_re,
        edges_3d=list(edges_3d_l)
    )
    net3d_l.organize_edges()
    return net3d_l
