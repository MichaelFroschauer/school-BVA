from abc import ABC, abstractmethod

class ImageFeatureBase(ABC):

    def __init__(self):
        self.description = ""

    @staticmethod
    @abstractmethod
    def CalcFeatureVal(imgRegion, FG_val):
        pass