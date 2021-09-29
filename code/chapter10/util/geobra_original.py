
import torch
from typing import Union, List, Optional, Set

# type specification
ScalorType = Union[float, torch.Tensor]
LengthType = Union[float, torch.Tensor]
SpaceType = Union[float, torch.Tensor]

__all__ = ["ScalorType", "LengthType", "SpaceType",
           "Origin",
           "Point", "Vector", "Edge", "Face", "Util"]


# -----------------------            geo             ----------------------------- #
class Point(object):
    def __init__(self, x: ScalorType, y: ScalorType = 0., z: ScalorType = 0., name: str = None):
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def __eq__(self, other):
        if type(self) != type(other): return False
        return self.x == other.x and self.y == other.y and self.z == other.z


Origin = Point(0., 0., 0.)


class Vector(object):
    def __init__(self, end: Point, start: Point = Origin):
        self.x = end.x - start.x
        self.y = end.y - start.y
        self.z = end.z - start.z
        self.start = start

    @property
    def length(self) -> LengthType:
        # the length of the vector
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5


class Edge(object):
    def __init__(self, v1: Point, v2: Point):
        self.v1 = v1
        self.v2 = v2
        self.name: Set = {v1.name, v2.name}
        self.info = None

    def __eq__(self, other):
        if type(self) != type(other): return False
        return self.name == other.name

    @property
    def length(self) -> LengthType:
        # the length of the edge
        return ((self.v1.x - self.v2.x) ** 2 + (self.v1.y - self.v2.y) ** 2 + (self.v1.z - self.v2.z) ** 2) ** 0.5


class Face(object):
    def __init__(self, v1: Point, v2: Point, v3: Point):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    @property
    def space(self) -> SpaceType:
        # the space of the Face
        vector1 = Vector(start=self.v1, end=self.v2)
        vector2 = Vector(start=self.v1, end=self.v3)
        return 0.5 * Util.cross(vector1, vector2).length



class Util(object):

    # --------------------    method for points   ------------------- #

    @staticmethod
    def distance(point1: Point, point2: Point, p=2):
        return ((point1.x - point2.x) ** p + (point1.y - point2.y) ** p + (point1.z - point2.z) ** p) ** (1 / p)

    # ----------------------- method for vecctors    ------------------ #
    @staticmethod
    def dot(vector1: Vector, vector2: Vector) -> ScalorType:
        # the dot product of 2 vectors
        return vector1.x * vector2.x + vector1.y + vector2.y + vector1.z + vector2.z

    @staticmethod
    def cross(vector1: Vector, vector2: Vector, start: Point = Origin) -> Vector:
        # the cross product of 2 vectors
        # vector1 x vector2
        vx = vector1.y * vector2.z - vector1.z * vector2.y
        vy = vector1.z * vector2.x - vector1.x * vector2.z
        vz = vector1.x * vector2.y - vector1.y * vector2.x
        end = Point(vx + start.x, vy + start.y, vz + start.z)
        return Vector(start=start, end=end)
