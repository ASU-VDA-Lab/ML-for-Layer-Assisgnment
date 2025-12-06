from data_struct_util import Net2D, Net3D
import statistics

import statistics

class RobustNormalizer:
    def __init__(self, clamp_k=3.0):
        self.clamp_k = clamp_k
        self._D, self._V, self._C = [], [], []
        self._fit = False

    def observe(self, D_e, V_e, C_e):
        self._D.append(float(D_e))
        self._V.append(float(V_e))
        self._C.append(float(C_e))

    def _robust_scale(self, xs):
        if not xs:
            return 0.0, 1.0
        m = statistics.median(xs)
        qs = statistics.quantiles(xs, n=20) 
        q05, q95 = qs[0], qs[-1]
        s = max(q95 - q05, 1e-12)
        return m, s

    def fit(self):
        self.D0, self.Ds = self._robust_scale(self._D)
        self.V0, self.Vs = self._robust_scale(self._V)
        self.C0, self.Cs = self._robust_scale(self._C)
        self._fit = True

    def _nz(self, x, m, s):
        z = (x - m) / s
        k = self.clamp_k
        return max(min(z, k), -k)

    def norm_D(self, x): return self._nz(x, self.D0, self.Ds)
    def norm_V(self, x): return self._nz(x, self.V0, self.Vs)
    def norm_C(self, x): return self._nz(x, self.C0, self.Cs)


metal_naming = {
    "asap7": lambda l: f"M{l}",
    "nangate45": lambda l: f"metal{l}",
    }
def write_net_info_to_file(filename, net_dict, is_3d=True):
    with open(filename, 'w') as f:
        for net_name, net in net_dict.items():
            if is_3d:
                f.write(f"Net: {net_name}\n")
                f.write(f"Source: {net.source}, Cap: {net.source.load_cap}, type: {net.source.node_type}, Fixed Layers: {net.source.fixed_layers}\n")
            
                for i, sink in enumerate(net.sinks):
                    f.write(f"Sink {i+1}: {sink}, Cap: {sink.load_cap}, type: {sink.node_type}, Fixed Layers: {sink.fixed_layers}\n")
                f.write(f"net's delay is {net.delay}\n")
                f.write(f"3D Edges: {len(net.edges_3d)}\n")
                all_3d_edges = set(net.edges_2d_with_layers).union(set(net.via_edges))
                for ed in all_3d_edges:
                    f.write(f"{ed.node1.coord}, {ed.node1.node_type}, to {ed.node2.coord}, {ed.node2.node_type} on {ed.layer1} and {ed.layer2}\n" )
                f.write("both should be same")
                for node1, node2, layer1, layer2 in net.edges_3d:
                    f.write(f"{node1.coord}, {node1.node_type}, to {node2.coord}, {node2.node_type} on {layer1} and {layer2}\n" )
                    
            else:
                f.write(f"Net: {net_name}\n")
                if len(net.edges_2d) == 0:
                    f.write(f"net.net_name has only via edges..please confirm")
                    continue
                f.write(f"Source: {net.source}, Cap: {net.source.load_cap}, type: {net.source.node_type}, Fixed Layers: {net.source.fixed_layers}\n")
            
                for i, sink in enumerate(net.sinks):
                    f.write(f"Sink {i+1}: {sink}, Cap: {sink.load_cap}, type: {sink.node_type}, Fixed Layers: {sink.fixed_layers}\n")

                f.write(f"2D Edges: {len(net.edges_2d)}\n")
                all_nodes = set(edge.node1 for edge in net.edges_2d).union(set(edge.node2 for edge in net.edges_2d))
                f.write(f"writing node info....\n")
                for n in all_nodes:
                    f.write(f"{n.coord}, subtree_cap : {n.subtree_cap}, node_type : {n.node_type}, load_cap : {n.load_cap}\n")
                src = None
                for node in all_nodes:
                    if node.node_type == 'source':
                        src = node
                src_edges = src.edges


                f.write(f"num of src edges is {len(src_edges)}\n")
                f.write(f"writing source edges..\n")

                for ed in src_edges:
                    f.write(f"{ed.node1.coord}, {ed.node1.node_type}, {ed.node1.fixed_layers} to {ed.node2.coord}, {ed.node2.node_type}, {ed.node2.fixed_layers} on {ed.layer1} and {ed.layer2}\n" )
                
                f.write(f"writing node's children...\n")
                for node in all_nodes:
                    children = node.children
                    f.write(f"node is {node} with type {node.node_type}\n")
                    f.write(f"i have children of {len(node.children)}\n")
                    for ch, R, C in children:
                        f.write(f"child node is {ch} with R is {R} and C is {C}\n")
                    f.write(f"node's parent is {node.parent_edge}\n")
                f.write(f"wiriting all edges..\n")


                for ed in net.edges_2d:
                    f.write(f"{ed.node1.coord}, {ed.node1.node_type}, {ed.node1.fixed_layers} to {ed.node2.coord}, {ed.node2.node_type}, {ed.node2.fixed_layers} on {ed.layer1} and {ed.layer2}\n" )
            f.write("\n")

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

def convert_net2d_to_net3d(net2d):
    edges_3d_l = set()
    visited_l = set()
    all_nodes_l = set(edge_l.node1 for edge_l in net2d.edges_2d).union(set(edge_l.node2 for edge_l in net2d.edges_2d))
    
    for n_l in all_nodes_l:
        n_l.edges.clear()

    for edge_l in net2d.edges_2d:
        edge_l.node1.add_edge(edge_l.node2, edge_l.layer1, edge_l.layer2)
        edge_l.node2.add_edge(edge_l.node1, edge_l.layer1, edge_l.layer2)
        e_key_l = (edge_l.node1, edge_l.node2, edge_l.layer1, edge_l.layer2)
        if e_key_l not in visited_l:
            visited_l.add(e_key_l)
            edges_3d_l.add(e_key_l)

    def add_vias_between(nod, lo, hi):
        """Adds via edges between layers lo to hi for a single node."""
        for l in range(lo, hi):
            via_edge_l = (nod, nod, l, l + 1)
            nod.add_edge(nod, l, l+1)
            if via_edge_l not in visited_l:
                visited_l.add(via_edge_l)
                edges_3d_l.add(via_edge_l)

    for node_l in all_nodes_l:
        connected_layers = [e.layer1 for e in node_l.get_all_edges() if e.layer1 is not None]
        if not connected_layers:
            continue

        min_l, max_l = min(connected_layers), max(connected_layers)

        if max_l != min_l:
            add_vias_between(node_l, min_l, max_l)

        # If no fixed_layers, assume default is layer 1
        fixed_layers = node_l.fixed_layers if node_l.fixed_layers else [1]

        for fixed_l in fixed_layers:
            for lay in connected_layers:
                if fixed_l != lay:
                    lo, hi = min(fixed_l, lay), max(fixed_l, lay)
                    add_vias_between(node_l, lo, hi)   

    all_nodes_re = set(ed.node1 for ed in net2d.edges_2d).union(set(ed.node2 for ed in net2d.edges_2d))
    src = None
    sinks = []
    all_src_sink_nodes_r= []
    for nod in all_nodes_re:
        if nod.node_type == 'source':
            src = nod
            all_src_sink_nodes_r.append(src)
            
        if nod.node_type == 'sink':
            sinks.append(nod)
            all_src_sink_nodes_r.append(nod)
    
    
    net3d_l = Net3D(
        net_name=net2d.net_name,
        src_sink_nodes=all_src_sink_nodes_r,
        edges_3d=list(edges_3d_l),
        num_pins=net2d.num_pins,
        net_slack=net2d.slack
    )
    net3d_l.source = src
    net3d_l.sinks = sinks
    net3d_l.organize_edges()
    return net3d_l


def convert_net2d_to_net3d_fixed(net2d, net_dict_fixed_3d): 
    edges_3d_re = set()
    visited_re = set()
    via_edges_3d = net_dict_fixed_3d[net2d.net_name].via_edges #keep the original via edges

    all_nodes_2d = set(edg.node1 for edg in net2d.edges_2d).union(set(edg.node2 for edg in net2d.edges_2d))
    
    for n in all_nodes_2d:
        n.edges.clear()

    for e in via_edges_3d:
        e.node1.add_edge(e.node2, e.layer1, e.layer2)
        e.node2.add_edge(e.node1, e.layer1, e.layer2)
        via_e_key =(e.node1, e.node2, e.layer1, e.layer2)
        if via_e_key not in visited_re:
            visited_re.add(via_e_key)
            edges_3d_re.add(via_e_key)

    for ed in net2d.edges_2d:
        ed.node1.add_edge(ed.node2, ed.layer1, ed.layer2)
        ed.node2.add_edge(ed.node1, ed.layer1, ed.layer2)
        e_key = (ed.node1, ed.node2, ed.layer1, ed.layer2)
        if e_key not in visited_re:
            visited_re.add(e_key)
            edges_3d_re.add(e_key)

    def add_vias_between(nod, lo, hi):
        for l in range(lo, hi):
            via_edge = (nod, nod, l, l + 1)
            if via_edge not in visited_re:
                visited_re.add(via_edge)
                edges_3d_re.add(via_edge)

    for n in all_nodes_2d: 
        connected_layers = [e_c.layer1 for e_c in n.get_all_edges() if e_c.layer1 is not None]
        if not connected_layers:
            continue

        min_l, max_l = min(connected_layers), max(connected_layers)

        if max_l != min_l:
            add_vias_between(n, min_l, max_l)

        fixed_layers_n = n.fixed_layers if n.fixed_layers else [1]

        for fixed_lay in fixed_layers_n:
            for lay in connected_layers:
                if fixed_lay != lay:
                    lo, hi = min(fixed_lay, lay), max(fixed_lay, lay)
                    add_vias_between(n, lo, hi)   
    
    
    all_nodes_re = set(edge_r.node1 for edge_r in net2d.edges_2d).union(set(edge_r.node2 for edge_r in net2d.edges_2d))
    src = None
    sinks = []
    all_src_sinks = []
    for node_re in all_nodes_re:
        if node_re.node_type == 'source':
            src = node_re
            all_src_sinks.append(src)
            
        if node_re.node_type == 'sink':
            sinks.append(node_re)
            all_src_sinks.append(node_re)
    
    
    net3d_re = Net3D(
        net_name=net2d.net_name,
        src_sink_nodes=all_src_sinks,
        edges_3d=list(edges_3d_re),
        num_pins=net2d.num_pins,
        net_slack=0
    )
    net3d_re.source = src
    net3d_re.sinks = sinks
    net3d_re.organize_edges()
    return net3d_re
    