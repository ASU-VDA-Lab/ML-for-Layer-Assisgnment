import math
import matplotlib.pyplot as plt
import torch

layer_configs = {
    0 : ('H', 0, 0, 0, 0),
    1 : ('V', 15, 7.04175E-02, 1e-10),        #direction, capacity per gcell grid, resistance, capacitance  per unit length
    2 : ('H', 13, 4.62311E-02, 1.84542E-01),
    3 : ('V', 12, 3.63251E-02, 1.53955E-01),
    4 : ('H', 12, 2.03083E-02, 1.89434E-01),
    5 : ('V', 3, 1.93005E-02, 1.71593E-01),
    6 : ('H', 2, 1.18619E-02, 1.76146E-01),
    7 : ('V', 2, 1.25311E-02, 1.47030E-01)
}
VIA_RESISTANCE = 200
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
def get_layers_by_dir(direction):
    if direction == 0:
        return [2,0,4,0,6,0]
    elif direction == 1:
        return [0,3,0,5,0,7]
class Node: 
    def __init__(self, x, y,  pin_name = None,node_type = "steiner", is_pin = False, fixed_layers = None):
        self.coord = (int(x), int(y))
        self.name = f"{self.coord[0]},{self.coord[1]}"
        self.pin_name = pin_name
        self.is_pin = is_pin
        self.node_type = node_type #type is src, sink or steiner default is steiner
        self.fixed_layers = fixed_layers if fixed_layers else []
        self.edges = []
        self.cap = 0.0
        self.slack_rise = 0.0
        self.slack_fall = 0.0
        self.via_count_flag = 0.0
        

    def add_edge(self, neighbor, layer1, layer2):
        if self.coord == neighbor.coord:
            direction = get_via_direction(self,neighbor,layer1, layer2)
        else:
            direction = get_direction_by_layer(self,neighbor,layer1)
        new_edge= Edge(self, neighbor, layer1, layer2, direction)
        if new_edge not in self.edges:
            self.edges.append(new_edge)
        if new_edge not in neighbor.edges:
            neighbor.edges.append(new_edge)
    @property
    def update_via_count_flag(self):
        num_H = 0
        num_V = 0
        for ed in self.edges:
            if ed.node1.coord[0] == ed.node2.coord[0]:
                num_V += 1
            else:
                num_H += 1
        if num_H == 0 or num_V == 0:
            self.via_count_flag = 0
        else:
            self.via_count_flag = max(num_V, num_H)

    def __repr__(self):
        return f"Node({self.coord}, pin={self.is_pin})"
    def __eq__(self, other):
        return (self.coord == other.coord)
    def __hash__(self):
        return hash(self.coord)
    def get_all_edges(self):
        """All edges connected to this node."""
        return self.edges
    def get_via_edges(self):
        return set(ed for ed in self.edges if ed.layer1 != ed.layer2)
        
    def get_flat_edges(self):
        return set(ed for ed in self.edges if ed.layer1 == ed.layer2)
    def get_feature_vector(self):
        return [
        self.coord[0],
        self.coord[1],
        self.cap,
        self.via_count_flag,
        float(self.is_pin)
        ]
def my_max(arr):
    if not arr:
        return 0
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val
class Edge:
    def __init__(self, node1, node2, layer1, layer2, direction = None):
        self.node1 = node1
        self.node2 = node2
        self.direction = direction
        self.edge_type = "via" if layer1 != layer2 else "flat"
        self.layer1 = layer1 if layer1 else None
        self.layer2 = layer2 if layer2 else None
        self.net_slack = 0.0
        self.num_pins = 0.0 
        self.length = math.sqrt((node1.coord[0]-node2.coord[0])**2 + (node1.coord[1]-node2.coord[1])**2)
        self.net_length = 0.0
        self.edge_direction_val = 1 if node1.coord[0] == node2.coord[0] else 0 # for 0 for H, 1 for V
        self.capacity_list = [layer_configs[l][1] for l in get_layers_by_dir(self.edge_direction_val)]
        self.res_list =  [self.length * layer_configs[l][2] for l in get_layers_by_dir(self.edge_direction_val)]
        self.cap_list = [self.length * layer_configs[l][3] for l in get_layers_by_dir(self.edge_direction_val)]
        self.via_res = self.get_via_res()
        self.actual_layer = layer1 
        self.predicted_layer = layer1 

    def __eq__(self, other):
        return (
            isinstance(other, Edge) and
            frozenset([self.node1.coord, self.node2.coord]) ==
            frozenset([other.node1.coord, other.node2.coord]) and
            self.layer1 == other.layer1 and self.layer2 == other.layer2
        )
    def get_via_res(self):
        fl1 = self.node1.fixed_layers
        fl2 = self.node2.fixed_layers
        max_fl1 = my_max(fl1)
        max_fl2 = my_max(fl2)
        return abs(max_fl1 - max_fl2) * VIA_RESISTANCE
    def __hash__(self):
        return hash((frozenset([self.node1.coord, self.node2.coord]), self.layer1, self.layer2)) # A, B == B, A
    def __repr__(self):
        return f"Edge({self.node1.coord} <-> {self.node2.coord}, length={self.length}, num_pins={self.num_pins})"
    
    def get_feature_vector(self):
        return [
            #self.num_pins,
            #self.length,
            self.net_slack,
            *self.res_list,
            *self.cap_list,
            #*self.capacity_list
        ]
  
  
class Net3D:
    def __init__(self, net_name, edges_3d=None, nodes=None):
        self.net_name = net_name
        self.edges_3d = edges_3d
        self.nodes = nodes
        self.edges_2d_with_layers = []
        self.via_edges = []
    def organize_edges(self):
        for edg in self.edges_3d:
            node1 = edg.node1
            node2 = edg.node2
            layer1 = edg.layer1
            layer2 = edg.layer2
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
    def __repr__(self):
        return f"Net({self.net_name}, nodes={len(self.nodes.keys())}, edges={len(self.edges_3d)})"

        

class CoreGrid3D():
    def __init__(self, die_gridx, die_gridy, grid_size_x, grid_size_y, num_layers):
        self.rows = math.ceil(die_gridx/grid_size_x)
        self.cols= math.ceil(die_gridy/grid_size_y)
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.num_layers = num_layers
        self.layer_names = list(range(2, num_layers+1)) 
        self.layer_to_index = {layer_id: idx for idx, layer_id in enumerate(self.layer_names)}
        self.usage = [[[0 for _ in self.layer_names] for _ in range(self.cols)] for _ in range(self.rows)]
        self.capacity = [[[layer_configs[layer][1] for layer in self.layer_names] for _ in range(self.cols)] for _ in range(self.rows)]
        self.cong_ratio = [[[0 for _ in self.layer_names] for _ in range(self.cols)] for _ in range(self.rows)] #update later after going through all nets
    
    def update_congestion(self):
        for row in range(self.rows):
            for col in range(self.cols):
                for lidx in range(len(self.layer_names)):
                    cap = self.capacity[row][col][lidx]
                    use = self.usage[row][col][lidx]
                    self.cong_ratio[row][col][lidx] = use / cap if cap > 0 else 0
        
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
    
    def update_usage(self, net): 
        if net.edges_3d == []:
            return 
        for edge in net.edges_3d:
            x1, y1 = edge.node1.coord
            x2, y2 = edge.node2.coord
            edge_dir = edge.edge_direction_val
            layer_list = get_layers_by_dir(edge_dir)
            
            points = self.get_grid_cells_crossed(x1, y1, x2, y2)
            for l in layer_list:
                if l < 2:
                    continue
                for r, c in points:
                    idx = self.layer_to_index[l]
                    self.usage[r][c][idx] += 1

    def get_congestion_map(self, layer_id):
        idx = self.layer_to_index[layer_id]
        return [[self.cong_ratio[r][c][idx] for c in range(self.cols)] for r in range(self.rows)]
    @property
    def congestion_image(self):
        return torch.tensor(self.cong_ratio, dtype=torch.float).permute(2, 0, 1)  # shape: [H, W, L]

def plot_congestion_map(congestion_map, layer_id):
    plt.figure(figsize=(6, 6))
    plt.imshow(congestion_map, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(label='Congestion Ratio')
    plt.title(f"Congestion Map - Layer {layer_id}")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"congestion_layer_{layer_id}.png", dpi=300)