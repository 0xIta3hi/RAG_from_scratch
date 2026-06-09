import random
import math

def cosine_similarity(v1:list[float], v2:list[float]) -> list[float]:
    dot_product = math.sumprod(v1,v2)
    v1_mag = math.hypot(v1)
    v2_mag = math.hypot(v2)
    cos_sim = dot_product / (v1_mag * v2_mag)
    return cos_sim

