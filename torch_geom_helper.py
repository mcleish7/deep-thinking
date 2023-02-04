import sys
if __name__ == '__main__':
    try:
        import torch
    except ImportError:
        print("""
# Please install PyTorch first, cf. https://pytorch.org/get-started/locally/
pip3.9 install torch
        """)
        quit(-1)

    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    gpu = torch.cuda.is_available()
    print(f"""\
# GPU available: {gpu}
# Versions
# Torch: {torch_version}
# CUDA : {cuda_version}
# CudNN: {cudnn_version}
""")

    TORCH = torch_version.split("+")[0]
    CUDA = "cu" + cuda_version.replace(".", "") if gpu else "cpu"

    print(f"""\
    
pip3.9 install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
pip3.9 install torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
pip3.9 install torch-cluster -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
pip3.9 install torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
pip3.9 install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip3.9 install torch-geometric
""")

# pip3.93.9 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cu102.html
# pip3.9 install torch-sparse -f https://data.pyg.org/whl/torch-1.10.2+cu102.html
# pip3.9 install torch-cluster -f https://data.pyg.org/whl/torch-1.10.2+cu102.html
# pip3.9 install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.2+cu102.html
# pip3.9 install torch-geometric

# pip3.9 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cpu.html
# pip3.9 install torch-sparse -f https://data.pyg.org/whl/torch-1.10.2+cpu.html
# pip3.9 install torch-cluster -f https://data.pyg.org/whl/torch-1.10.2+cpu.html
# pip3.9 install pyg-lib -f https://data.pyg.org/whl/torch-$1.10.2+$cpu.html -- doesn't work
# pip3.9 install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.2+cpu.html
# pip3.9 install torch-geometric
sys.exit()
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='.', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# model = GCN(dataset.num_features, 16, dataset.num_classes)
# print("complete")