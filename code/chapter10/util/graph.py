
import jax
import jax.numpy as jnp
import numpy as np

import logging
from typing import Union, List, Optional, Set, Bool


# 类型指定
ScalorType = Union[float, np.ndarray, jax.ndarray]
LengthType = Union[float, np.ndarray, jax.ndarray]
SpaceType  = Union[float, np.ndarray, jax.ndarray]


class Point(object):
    def __init__(self, x: Optional[ScalorType], 
                       y: Optional[ScalorType] = 0., 
                       z: Optional[ScalorType] = 0., 
                       name: str = None):
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def __eq__(self, other):
        # 如果 P1 和 P2 都是 Point类，且名称（也就是name参数）相同，我们就认为两个点是相同的， 
        #     ———— 也就是说 P1 == P2 将会返回 True
        if type(self) != type(other): return False
        return self.name == other.name
    
    def __str__(self):
        # 指定调用print函数时，程序终端将会输出的string
        return "class Point:\n\t{}:({},{},{})".format(self.name, self.x, self.y, self.z)

Origin = Point(0., 0., 0., name="Origin")

class VertexInfo(object):
    def __init__(self, index:int = -1,
                       vertex_init_pos  :np.ndarray = None,
                       vertex_ground_pos:np.ndarray = None):
        self.idx: int = index     # 储存该顶点在输入 3xN 顶点列表之中的位置
        self.working:Bool = False # 描述该节点是否处在工作区
        self.vertex_init_pos  : np.ndarray = vertex_init_pos   # 顶点的初始位置
        self.vertex_ground_pos: np.ndarray = vertex_ground_pos # 固定在地面上的促动器的坐标 

class Vertex(Point):
    def __init__(self, name: str, 
                          x: ScalorType = None, 
                          y: ScalorType = None, 
                          z: ScalorType = None):
        super().__init__(x, y, z, name = name)  # 初始化顶点的名称及位置参数
        self.adjacent: List[Vertex] = []
        self.info : VertexInfo = None

    def __eq__(self, other):
        # 如果Vertex的名称(name参数)相同，我们就认为两个Vertex是一样的
        if type(self) != type(other): return False
        if self.name == other.name:
            # 如果两个顶点的名称相同，我们将会检查这两个节点的坐标和相邻节点是否相同
            # 如若不同，将会在终端产生warning, 但程序并不报错
            if self.x != other.x or self.y != other.y or self.z != other.z:
                logging.warning("Inconsistent position between Vertexes {}:\n"
                                "  [self]: {}\n [other]: {}".format(self.name, (self.x, self.y, self.z),
                                                                   (other.x, other.y, other.z)))
            if self.adjacent != other.adjacent:
                logging.warning("Inconsistent adjacent list between Vertex {}:\n"
                                " [self ]: {}\n [other]: {}".format(self.name, self.adjacent, other.adjacent))
            return True
        return False
    
    def __str__(self):
        # 指定调用print函数时，程序终端将会输出的string
        return "class Vertex:\n\tname: {}\n\tposition: {}\n\tadjacent vertex: {}".\
            format(self.name, (self.x, self.y, self.z), [_.name for _ in self.adjacent])
    
    @property
    def n_adjacent(self):
        return len(self.adjacent)

class Edge(object):
    def __init__(self, v1: Vertex, v2: Vertex):
        self.v1 = v1
        self.v2 = v2
        self.name: Set = {v1.name, v2.name}

    def __eq__(self, other):
        if type(self) != type(other): return False
        return self.name == other.name

    @property
    def length(self) -> LengthType:
        # the length of the edge
        return ((self.v1.x - self.v2.x) ** 2 + (self.v1.y - self.v2.y) ** 2 + (self.v1.z - self.v2.z) ** 2) ** 0.5


class Graph(object):
    def __init__(self, vertex_list: Optional[List[Vertex]]=[], 
                       edge_list  : Optional[List[Edge  ]]=[]):
        self.vertex_list: List[Vertex] = vertex_list
        self.edge_list  : List[Edge  ] = [] if edge_list == None else edge_list

    def update_edge(self):
        logging.info("Updating Edges...")
        from tqdm import tqdm
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