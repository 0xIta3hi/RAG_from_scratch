import random
import math

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    # 1. calculate dot product
    dot_product = math.sumprod(v1,v2)
    # 2. calculate magnitude for v1
    v1_mag = math.hypot(*v1)
    # 3. calculate magnitude for v2
    v2_mag = math.hypot(*v2)
    # 4. return cosine similarity.
    if v1_mag == 0.0 or v2_mag == 0:
        return 0.0
    else:
        cos_sim = dot_product / (v1_mag * v2_mag)
        return cos_sim

class HNSW_Node:
    def __init__(self, node_id:int, vector:list[float], metadata:list, max_level:int):
        self.node_id = node_id
        self.vector = vector
        self.metadata = metadata
        # maps layer number -> list of neighbor node IDs  
        self.connections = {layer:[] for layer in range(max_level + 1)}

class HNSW_Index:
    def __init__(self, dimension: int = 384, M: int = 16, mL: float = 0.62):
        self.dimension = dimension
        self.M = M                # Max connections per node per layer
        self.mL = mL              # Factor for exponential layer selection
        
        self.nodes = {}           # Registry: node_id -> HNSWNode object
        self.entry_point_id = None 
        self.max_level = -1
        
    def _assign_random_level(self) -> int:
        # Get a number between 0,1
        r = random.uniform(0, 1)
        if r == 0: r == 0.001
        return int(-math.log(r) * self.mL)
    
    def _greedy_search_layer(self, query_vector:list[float], entry_node_id:int, layer_id:int) -> int:
        current_id = entry_node_id
        current_score = cosine_similarity(query_vector, self.nodes[current_id].vector)

        while True:
            changed = False
            neighbors = self.nodes[current_id].connections[layer_id]
            for neighbor_id in neighbors:
                neighbor_score = cosine_similarity(query_vector, self.nodes[neighbor_id].vector)
                if neighbor_score > current_score:
                    current_score = neighbor_score
                    current_id = neighbor_id
                    changed = True
            if not changed:
                continue
            return current_id
    
    def insert(self, node_id: int, vector: list[float], metadata: dict):
        # 1. Determine the birth level for this new node
        insert_level = self._assign_random_level()
        
        # 2. Instantiate the new node object
        new_node = HNSW_Node(node_id, vector, metadata, insert_level)
        self.nodes[node_id] = new_node
        
        # EDGE CASE: If this is the absolute first node in the database
        if self.entry_point_id is None:
            self.entry_point_id = node_id
            self.max_level = insert_level
            return

        # Start tracking our navigation entry point
        curr_obj_id = self.entry_point_id
        
        # PHASE 1: The Upper-Layer Fast Pass (Top Level down to Insert Level + 1)
        # We greedily fly across the top layers just to find the closest gateway node
        for layer in range(self.max_level, insert_level, -1):
            curr_obj_id = self._greedy_search_layer(vector, curr_obj_id, layer)
            
        # PHASE 2: The Lower-Layer Link-up (Insert Level down to Layer 0)
        # Now the node is actually born on these layers, so we must connect it!
        # We look at the layer we are currently on or the maximum birth level, whichever is lower
        start_layer = min(self.max_level, insert_level)
        
        for layer in range(start_layer, -1, -1):
            # Find the local entry point for this specific layer
            curr_obj_id = self._greedy_search_layer(vector, curr_obj_id, layer)
            
            # For simplicity in our custom build, we will make a direct bidirectional link
            # between the new node and the closest node found on this layer.
            # (In the full paper, they find top 'M' neighbors, but 1-to-1 tracking is perfect for our lab)
            
            # Establish the links
            self.nodes[curr_obj_id].connections[layer].append(node_id)
            new_node.connections[layer].append(curr_obj_id)
            
            # Enforce the Guardrail: Prune connections if they exceed self.M
            if len(self.nodes[curr_obj_id].connections[layer]) > self.M:
                # Simple pruning optimization: pop the first connection or sort by distance
                self.nodes[curr_obj_id].connections[layer].pop(0)

        # 3. System Upgrade Check: If the new node climbed higher than the old king
        if insert_level > self.max_level:
            self.entry_point_id = node_id
            self.max_level = insert_level

    def search(self, query_vector:list[float], k:int=3) -> list[dict]:
        if self.entry_point_id is None:
            return []
        
        curr_obj_id = self.entry_point_id
        
        for layer in range(self.max_level, 0, -1):
            curr_obj_id = self._greedy_search_layer(query_vector, curr_obj_id, layer)
        
        layer_0_neighbor = self.nodes[curr_obj_id].connections[0]

        candidates = list(set(layer_0_neighbor + [curr_obj_id]))

        scored_candidates = []

        for node_id in candidates:
            node = self.nodes[node_id]
            score = cosine_similarity(query_vector, node.vector)

            scored_candidates.append({
                "text":node.metadata["text"],
                "source":node.metadata["source"],
                "page":node.metadata["page"],
                "score":score
                })
        sorted_result = sorted(scored_candidates, key= lambda x:x["score"], reverse=True)
        return sorted_result
    
    


