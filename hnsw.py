import random
import math

def cosine_similarity(v1:list[float], v2:list[float]) -> list[float]:
    dot_product = math.sumprod(v1,v2)
    v1_mag = math.hypot(v1)
    v2_mag = math.hypot(v2)
    cos_sim = dot_product / (v1_mag * v2_mag)
    return cos_sim

class HNSW_Node:
    def __init__(self, node_id:int, vector:list[float], metadata:list, max_level:int):
        self.node_id = node_id
        self.vector = vector
        self.metadata = metadata
        # maps layer number -> list of neighbor node IDs  
        self.connections = {layer:[] for layer in range(max_level + 1)}

        
