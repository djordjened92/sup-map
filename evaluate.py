import os
import yaml
import glob
import torch
from model import GCN
from dataset import load_labels
from utils import *
from metrics import pairwise, bcubed
from torch_geometric.loader import NeighborLoader
from train import inference


model_dir = '/home/djordje/Documents/Projects/face-mod/checkpoints/bs64_k60_lr1e-4_ep400_do0.15_004'
config = yaml.safe_load(open(glob.glob(os.path.join(model_dir, '*.yaml'))[0], 'r'))

# Set torch device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Create model
model = GCN(in_dim=config['FEATURE_DIM'], hidden_dim=config['HIDDEN_DIM'], out_dim=config['OUT_DIM'], dropout=config['DROPOUT'])
print(f'GCN model:\n {model}\n')

ckpt_path = glob.glob(os.path.join(model_dir, '*.pth'))[0]
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt)
model = model.to(device)

# Input data graph
val_label_path = config['VAL_LABELS']
val_labels = load_labels(val_label_path)
val_classes = len(np.unique(val_labels))
print(f'num of labels: {val_labels.shape[0]}')
print(f'#cls: {val_classes}')

graph_dir = f'ws{config["WINDOW_SIZE"]}'
test_in_graph = torch.load(f'{graph_dir}/test_graph.pt')
test_nbrs_bounds = torch.load(f'{graph_dir}/test_nbrs_bounds.pt')
test_in_graph['nbrs_bounds'] = test_nbrs_bounds
test_nbrs = torch.load(f'{graph_dir}/test_nbrs.pt')
test_ji = torch.load(f'{graph_dir}/test_ji.pt')

test_loader = NeighborLoader(
    test_in_graph,
    num_neighbors=[-1, -1],
    batch_size=512,
    shuffle=False
)

# Inference
model.eval()
with torch.no_grad():
    # Evaluate model on the test graph, optimize tau
    print('Evaluate test graph:')
    test_sims = inference(test_in_graph,
                          test_nbrs,
                          test_ji,
                          model,
                          test_loader,
                          config,
                          device)

    pred_labels = face_cluster(test_nbrs.cpu().numpy(),
                               test_sims.cpu().numpy(),
                               flow_model='undirected',
                               silent=False)
    avg_pre_p, avg_rec_p, fscore_p = pairwise(val_labels, pred_labels)
    avg_pre_b, avg_rec_b, fscore_b = bcubed(val_labels, pred_labels)

    print('#pairwise: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre_p, avg_rec_p, fscore_p))
    print('#bicubic: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre_b, avg_rec_b, fscore_b))