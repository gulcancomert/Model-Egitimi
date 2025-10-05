import torch
import numpy as np
import cv2

# Attention Rollout for ViT (timm models)
class VitAttentionRollout:
    def __init__(self, model, head_fusion='mean', discard_ratio=0.0):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        self.hooks = []
        # Register hooks on each block's attention
        for blk in getattr(self.model, 'blocks', []):
            # hook on attn_drop input (which receives attn weights)
            h = blk.attn.attn_drop.register_forward_hook(self._hook)
            self.hooks.append(h)

    def _hook(self, module, input, output):
      
        if len(input) > 0:
            self.attentions.append(input[0].detach().cpu())

    def clear(self):
        self.attentions.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def __call__(self):
        # Stack layers: list of (B,H,T,T) -> (L,B,H,T,T)
        if not self.attentions:
            return None
        attns = torch.stack(self.attentions)  # (L,B,H,T,T)
        # fuse heads
        if self.head_fusion == 'mean':
            attns = attns.mean(dim=2)  # (L,B,T,T)
        elif self.head_fusion == 'max':
            attns = attns.max(dim=2).values
        else:
            attns = attns.mean(dim=2)
        # Attention rollout across layers
        eye = torch.eye(attns.size(-1)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        attns = attns + eye  # add residual
        attns = attns / attns.sum(dim=-1, keepdim=True)
        # cumulative matmul
        result = attns[0]
        for i in range(1, attns.shape[0]):
            result = result @ attns[i]
        # take the attention from CLS token to others: index 0 is CLS
        mask = result[:, 0, 1:]  # (B, T-1)
        return mask  # flattened patch attentions per image in batch

def attention_to_map(mask, grid_size, image_hw):
    # mask: (B, P), P = grid*grid
    b, p = mask.shape
    g = grid_size
    maps = []
    for i in range(b):
        m = mask[i].reshape(g, g).numpy()
        m = m / (m.max() + 1e-6)
        m = cv2.resize(m, (image_hw[1], image_hw[0]))  # (W,H) -> (H,W)
        maps.append(m)
    return maps
