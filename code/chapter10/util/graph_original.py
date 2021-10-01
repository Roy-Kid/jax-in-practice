
import torch
import logging
from typing import List,Union, Optional
from util.geobra import *
from tqdm import tqdm


__all__ = ["Vertex", "Graph"]


class Vertex(Point):
    def __init__(self, name: Union[int, str], x: ScalorType = None, y: ScalorType = None, z: ScalorType = None):
        super().__init__(x, y, z, name)
        self.adjacent: List[Vertex] = []
        self.info = None

    def __eq__(self, other):
        # if the vertexes' names are the same, then we regard them as identical
        # if the same name occur
        if not (type(self) == type(other)):
            return False
        if self.name == other.name:
            if self.x != other.x or self.y != other.y or self.z != other.z:
                logging.warning("Inconsistent position between Vertexes {}:\n"
                                "  [self]: {}\n [other]: {}".format(self.name, (self.x, self.y, self.z),
                                                                  (other.x, other.y, other.z)))
            if self.adjacent != other.adjacent:
                logging.warning("Inconsistent adjacent list between Vertex {}:\n"
                                " [self ]: {}\n [other]: {}".format(self.name, self.adjacent, other.adjacent))
            return True
        else:
            return False

    def __str__(self):
        x_str = self.x.item() if isinstance(self.x, torch.Tensor) else self.x
        y_str = self.y.item() if isinstance(self.y, torch.Tensor) else self.y
        z_str = self.z.item() if isinstance(self.z, torch.Tensor) else self.z

        return "class Vertex:\n\tname: {}\n\tposition: {}\n\tadjacent vertex: {}".\
            format(self.name, (x_str, y_str, z_str), [_.name for _ in self.adjacent])

    def add_adjacent(self, vertex):
        assert type(self) == type(vertex)
        if vertex not in self.adjacent:
            self.adjacent.append(vertex)
            # if self not in vertex.adjacent:
            #     vertex.adjacent.append(self)
            return True
        return False

    @property
    def n_adjacent(self):
        return len(self.adjacent)

class Graph(object):
    def __init__(self, vertex_list: Optional[List[Vertex]], edge_list: Optional[List[Edge]]=None):
        self.vertex_list: List[Vertex] = [] if vertex_list == None else vertex_list
        self.edge_list: List[Edge] = [] if edge_list == None else edge_list

    def update_edge(self):
        logging.info("updating Edges...")
        for v in tqdm(self.vertex_list):
            for v_adjacent in v.adjacent:
                assert v in v_adjacent.adjacent
                edge = Edge(v, v_adjacent)
                if edge not in self.edge_list:
                    self.edge_list.append(edge)

    @property
    def n_edge(self):
        return len(self.edge_list)

    @property
    def n_vertex(self):
        return len(self.vertex_list)




