import os
import torch
from rich.console import Console

console = Console()
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    console.print(f"[b]Size (MB)[/b]: {os.path.getsize('temp.p')/1e6}")
    os.remove('temp.p')

def compute_sparisty(model):
    total_weight = 0
    total_nelement = 0
    for m in model.named_parameters():
        name = m[0]
        tensor = m[1]
        w = float(torch.sum(tensor == 0))
        ne = float(tensor.nelement())
        # print(f"{name} : {sparsity}")
        total_weight += w
        total_nelement += ne
    sparsity = 100 * total_weight / total_nelement
    return sparsity