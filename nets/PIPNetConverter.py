import numpy as np


class Converter:
    def __init__(self):
        super().__init__()
        pass
    
    def compute_nme(self, output, target, norm):
        return np.mean(np.linalg.norm(output - target, axis=1)) / norm
    
    def __call__(self, output, target, index=[0, 9]):
        """ Compute NME """
        output = output.reshape(-1, 2).astype(np.float32)
        target = target.reshape(-1, 2).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(target[index[0]] - target[index[1]])
        return self.compute_nme(output, target, norm)
