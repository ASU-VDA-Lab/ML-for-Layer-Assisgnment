import math
layer_configs = {
    1 : ('V', 15, 7.04175E-02, 1e-10),        #direction, capacity per gcell grid, resistance, capacitance  per unit length for ASAP7
    2 : ('H', 13, 4.62311E-02, 1.84542E-01),
    3 : ('V', 12, 3.63251E-02, 1.53955E-01),
    4 : ('H', 12, 2.03083E-02, 1.89434E-01),
    5 : ('V', 3, 1.93005E-02, 1.71593E-01),
    6 : ('H', 2, 1.18619E-02, 1.76146E-01),
    7 : ('V', 2, 1.25311E-02, 1.47030E-01)
}
VIA_RESISTANCE = 200 #for ASAP7
'''
#for nangate45, uncomment this nad change num_layers=10
layer_configs = {
    1 : ('H', 15, 5.4286E-03, 7.41819E-02),       
    2 : ('V', 13, 3.5714E-03, 6.74606E-02),
    3 : ('H', 12, 3.5714E-03, 8.88758E-02),
    4 : ('V', 12, 1.5000E-03, 1.07121E-01),
    5 : ('H', 7, 1.5000E-03, 1.08964E-01),
    6 : ('V', 6, 1.5000E-03, 1.02044E-01),
    7 : ('H', 5, 1.8750E-04, 1.10436E-01),
    8 : ('V', 3, 1.8750E-04, 9.69714E-02),
    9 : ('H', 2, 3.7500E-05, 3.6864E-02),
    10 : ('V', 2, 3.7500E-05, 2.8042E-02),
}
VIA_RESISTANCE = 2.94
'''


def get_via_direction(node1, node2, layer1, layer2):
    if layer1 < layer2:
          return (node1.name, node2.name)
    else:
          return (node2.name, node1.name)

def get_direction_by_layer(node1, node2, layer):
    if layer == None:
        if node1.coord[0] == node2.coord[0]:
            return (
            (node1.name, node2.name)
            if node1.coord[1] <= node2.coord[1]
            else (node2.name, node1.name)
        )
        else:
            return (
            (node1.name, node2.name)
            if node1.coord[0] <= node2.coord[0]
            else (node2.name, node1.name)
        )
    else:
        direction_type = layer_configs[layer][0]
        if direction_type == "H":
            return (
            (node1.name, node2.name)
            if node1.coord[0] <= node2.coord[0]
            else (node2.name, node1.name)
        )
        elif direction_type == "V":
            return (
            (node1.name, node2.name)
            if node1.coord[1] <= node2.coord[1]
            else (node2.name, node1.name)
            )
        else:
            return None

class Node: 
    def __init__(self, x, y,pin_name=None, node_type = "steiner", fixed_layers=None):
        self.coord = (int(x), int(y))
        self.name = f"{self.coord[0]},{self.coord[1]}"
        self.pin_name = pin_name
        self.node_type = node_type #type is src, sink or steiner default is steiner
        self.fixed_layers = fixed_layers if fixed_layers else []
        self.edges = [] 
        self.parent_edge = None  
        self.children = []  
        self.load_cap = 0.0 
        self.subtree_cap = 0.0 
        

    def add_edge(self, neighbor, layer1, layer2):
        if self.coord == neighbor.coord:
            direction = get_via_direction(self,neighbor,layer1, layer2)
        else:
            direction = get_direction_by_layer(self,neighbor,layer1)
        new_edge= Edge(self, neighbor, layer1, layer2, direction)
        
        if new_edge not in self.edges:
            self.edges.append(new_edge)

    def __repr__(self):
        return f"Node({self.pin_name}, {self.coord}, type={self.node_type})"
    def __eq__(self, other):
        return (self.coord == other.coord)
    def __hash__(self):
        return hash(self.coord)
    
    def __str__(self):
        return f"Node(pin_name={self.pin_name}, x={self.coord[0]}, y={self.coord[1]}, type={self.node_type}, cap={self.load_cap})"

    def get_all_edges(self):
        """All edges connected to this node."""
        return self.edges
    def get_via_edges(self):
        return set(ed for ed in self.edges if ed.layer1 != ed.layer2)
        
    def get_flat_edges(self):
        return set(ed for ed in self.edges if ed.layer1 == ed.layer2)

class Edge:
    def __init__(self, node1, node2, layer1, layer2, direction = None):
        self.name = f"{node1,node2, layer1, layer2}"
        self.node1 = node1
        self.node2 = node2
        self.direction = direction
        self.edge_type = "via" if layer1 != layer2 else "flat"
        self.layer1 = layer1 if layer1 else None
        self.layer2 = layer2 if layer2 else None
        self.length = math.sqrt((node1.coord[0]-node2.coord[0])**2 + (node1.coord[1]-node2.coord[1])**2)
        self.resistance = (
            layer_configs[layer1][2] * self.length
            if layer1 == layer2 and layer1 is not None and layer2 is not None
            else VIA_RESISTANCE
            if layer1 != layer2 and layer1 is not None and layer2 is not None
            else 0.0
            )
        self.capacitance = layer_configs[layer1][3] * self.length if (layer1 == layer2 and layer1 != None and layer2 != None) else 0.0


    def __eq__(self, other):
        return (
            isinstance(other, Edge) and
            frozenset([self.node1.coord, self.node2.coord]) ==
            frozenset([other.node1.coord, other.node2.coord]) and
            self.layer1 == other.layer1 and self.layer2 == other.layer2
        )

    def __hash__(self):
        return hash((frozenset([self.node1.coord, self.node2.coord]), self.layer1, self.layer2)) # A, B == B, A
    def __repr__(self):
        return f"Edge({self.node1.coord} <-> {self.node2.coord}, layer1={self.layer1},layer2={self.layer2}, dir={self.direction})"
  
  
  
class Net3D:
    def __init__(self, net_name, src_sink_nodes, edges_3d = None, num_pins =None, net_slack=None): #when taking edges_3d just take a list
        self.net_name = net_name
        self.slack = net_slack 
        self.nodes = src_sink_nodes
        self.source = None
        self.sinks = []
        self.num_pins = num_pins
        self.edges_3d = edges_3d #(node1, node2, layer1, layer2)
        self.via_edges = []
        self.edges_2d_with_layers = []
        self.delay = None
        self.length = sum(e.length for e in self.edges_2d_with_layers)
        self.num_vias = len(self.via_edges)
        self.total_overflow = 0
        self.max_overflow = 0

    def organize_edges(self):
        for (node1, node2, layer1, layer2) in self.edges_3d:
            if layer1 != layer2:
                direction = get_via_direction(node1, node2, layer1, layer2)
                if direction == (node2.name, node1.name):
                    via_edge = Edge(node2,node1, layer2, layer1, direction)
                else:
                    via_edge = Edge(node1,node2, layer1, layer2, direction)
                if via_edge not in self.via_edges:
                    self.via_edges.append(via_edge)
            else:
                direction = get_direction_by_layer(node1, node2,layer1)
                if direction == (node2.name, node1.name):
                    ed = Edge(node2, node1, layer2, layer1, direction)
                else:
                    ed = Edge(node1, node2, layer1, layer2, direction)
        
                if ed not in self.edges_2d_with_layers:
                    self.edges_2d_with_layers.append(ed)


class Net2D:
    def __init__(self, net_name, edges_2d, source, sinks, num_pins, slack):
        self.net_name = net_name
        self.num_pins = num_pins
        self.source = source
        self.sinks = sinks
        self.slack = slack
        self.edges_2d = edges_2d
        self.length = sum(e.length for e in self.edges_2d)
        self.total_overflow = 0
        self.max_overflow = 0
        

class CoreGrid3D():
    def __init__(self, die_gridx, die_gridy, grid_size_x, grid_size_y, num_layers):
        self.rows = math.ceil(die_gridx/grid_size_x)
        self.cols= math.ceil(die_gridy/grid_size_y)
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.num_layers = num_layers
        self.layer_names = [l for l in sorted(layer_configs.keys()) if l != 1][:num_layers]
        self.layer_index_map = {layer: i for i, layer in enumerate(self.layer_names)}
        self.usage = [[[0 for _ in self.layer_names] for _ in range(self.cols)] for _ in range(self.rows)]
        self.usage_2d = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.capacity = [[[layer_configs[layer_name][1] for layer_name in self.layer_names] for _ in range(self.cols)] for _ in range(self.rows)]
        self.overflow = [[[0 for _ in self.layer_names] for _ in range(self.cols)] for _ in range(self.rows)]

    def coord_to_cell(self, x, y):
        x = min(max(x, 0), self.grid_size_x * self.rows - 1)
        y = min(max(y, 0), self.grid_size_y * self.cols - 1)
        col = int(x // self.grid_size_x)
        row = int(y // self.grid_size_y)
        return max(0, min(row, self.rows - 1)), max(0, min(col, self.cols - 1))

    def get_grid_cells_crossed(self, x1, y1, x2, y2):
        x1_idx, y1_idx = self.coord_to_cell(x1, y1)[1], self.coord_to_cell(x1, y1)[0]
        x2_idx, y2_idx = self.coord_to_cell(x2, y2)[1], self.coord_to_cell(x2, y2)[0]
        points = []
        dx = abs(x2_idx - x1_idx)
        dy = abs(y2_idx - y1_idx)
        x, y = x1_idx, y1_idx
        sx = 1 if x1_idx < x2_idx else -1
        sy = 1 if y1_idx < y2_idx else -1
        err = dx - dy
        while True:
            if 0 <= y < self.rows and 0 <= x < self.cols:
                points.append((y, x))
            if x == x2_idx and y == y2_idx:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points