import os
import yaml
import glob
import torch
import shutil
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import GCN, avg_internal_variance, randomwalk_nodes, filter_graph_by_rw, map_eq_loss
from utils import *
from dataset import load_feat, load_labels, FeatureDataset
from metrics import pairwise, bcubed
from negatives_sampler import EntropyNegativeSampler
from torch_scatter import scatter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pytorch_metric_learning import samplers
from torch_geometric.loader import NeighborLoader, RandomNodeLoader
import torch_geometric.transforms as T

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"

# Manual seed
seed = 123
set_random_seed(seed)

class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data_loader)

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)  # Reset the data loader
            data = next(self.data_iter)
        return data

def train(model,
          optimizer,
          epoch,
          num_epochs,
          neg_sampler,
          training_loader,
          iterations,
          config,
          device):
    TAU = config.get('tau', 0.5)
    # We reduce the final multiplier because these edges are "high quality"
    # But we sample a larger pool initially to find them.
    NEG_MULTIPLIER = config.get('neg_multiplier', 5.0) 
    POOL_FACTOR = 50.0 # How many random pairs to check before picking the hardest
    
    two_hop_transform = T.TwoHop()
    running_loss = 0.

    for i in tqdm(range(iterations), 'Train loader:'):
        data = next(training_loader)
        data = data.to(device)
        data = two_hop_transform(data)
        b_labels = data.y
        
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        # 1. Base edges (Positive structure)
        base_edges = data.edge_index
        num_real_edges = base_edges.size(1)

        # # -----------------------------------------------------------------
        # # 2. Hard Negative Mining Stage - simple topk sampler
        # # -----------------------------------------------------------------
        # # A. Sample a large pool of random pairs
        # num_pool = int(num_real_edges * POOL_FACTOR)
        # pool_indices = torch.randint(0, data.num_nodes, (2, num_pool), device=device)
        
        # # B. Filter for actual negatives (different labels)
        # is_neg_label = (b_labels[pool_indices[0]] != b_labels[pool_indices[1]])
        # pool_neg_indices = pool_indices[:, is_neg_label]
        
        # if pool_neg_indices.size(1) == 0:
        #     continue

        # # C. Find "Hard" Negatives: Calculate similarity for the pool
        # # We use .detach() if we only want to use them as indices, 
        # # but keeping them attached is fine for map_eq_loss.
        # row_p, col_p = pool_neg_indices
        # pool_sims = (out[row_p] * out[col_p]).sum(dim=-1)
        
        # # D. Select Top-K hardest (highest similarity)
        # num_hard = int(num_real_edges * NEG_MULTIPLIER)
        # num_hard = min(num_hard, pool_neg_indices.size(1)) # Safety check
        
        # _, topk_idx = torch.topk(pool_sims, k=num_hard)
        # neg_edge_index = pool_neg_indices[:, topk_idx]

        # -----------------------------------------------------------------
        # 2. Negative Mining Stage - Entropy-based
        # -----------------------------------------------------------------
        neg_edge_index = neg_sampler.sample(out=out,
                                            edge_index=base_edges,
                                            labels=b_labels,
                                            epoch=epoch,
                                            num_epochs=num_epochs)

        if neg_edge_index is not None:
            filtered_edge_index = torch.cat([base_edges, neg_edge_index], dim=1)
        else:
            filtered_edge_index = base_edges

        # -----------------------------------------------------------------
        # 3. Loss
        # -----------------------------------------------------------------
        avg_var, var_means = avg_internal_variance(base_edges, out, b_labels)
        
        # Map Equation Loss + Geometric Regularization
        curr_loss = map_eq_loss(out, filtered_edge_index, b_labels, tau=TAU) + 0.2 * avg_var + var_means
        curr_loss.backward()
        optimizer.step()
        
        running_loss += curr_loss.item()
        torch.cuda.empty_cache()

    return running_loss / iterations

def inference(in_graph, model, data_loader, config, device):
    out_x = torch.zeros((in_graph.num_nodes, config['OUT_DIM']), device=device)
    for data in data_loader:
        data = data.to(device)
        local_idx = (data.input_id[:, None]==data.n_id).nonzero()[:, 1]
        out_x[data.input_id, :] = model(data.x, data.edge_index)[local_idx]

    return out_x

def main(config_path, device):
    # Load config
    config = yaml.safe_load(open(config_path, 'r'))
    k = config['K']

    # Load data
    feat_dim = config['FEATURE_DIM']

    train_label_path = config['TRAIN_LABELS']
    train_labels = load_labels(train_label_path)
    print(f'num of labels: {train_labels.shape[0]}')
    print(f'#cls: {len(np.unique(train_labels))}\n')

    graph_dir = f'ws{config["WINDOW_SIZE"]}_k45'

    if os.path.exists(graph_dir):
        test_in_graph = torch.load(f'{graph_dir}/test_graph.pt', weights_only=False)
        train_in_graph = torch.load(f'{graph_dir}/train_graph.pt', weights_only=False)
    else:
        os.makedirs(graph_dir)

        # Generate train graph
        train_feat_path = config['TRAIN_FEATURES']
        train_feat = load_feat(train_feat_path, feat_dim)
        train_features = l2norm(train_feat)
        print('Train:')
        print('features shape:', train_features.shape)
        train_in_graph, _ = gen_graph(torch.from_numpy(train_features).to(device),
                                                        config,
                                                        45,
                                                        device,
                                                        z_score=False)
        # train_in_graph = preprocess_graph_deterministic(train_in_graph,
        #                                      batch_size=10,
        #                                      device="cpu",
        #                                      batching="range",
        #                                      write_dir=None,
        #                                      pe_type="spd",
        #                                      pe_kwargs={"num_anchors": 3, "cutoff": 4})
        torch.save(train_in_graph, f'{graph_dir}/train_graph.pt')

        # Generate test graph 
        val_feat_path = config['VAL_FEATURES']
        val_feat = load_feat(val_feat_path, feat_dim)
        val_features = l2norm(val_feat)
        print('Test:')
        print('features shape:', val_features.shape)
        test_in_graph, _ = gen_graph(torch.from_numpy(val_features).to(device),
                                                        config,
                                                        45,
                                                        device,
                                                        z_score=False)
        # test_in_graph = preprocess_graph_deterministic(test_in_graph,
        #                                 batch_size=10,
        #                                 device="cpu",
        #                                 batching="range",
        #                                 write_dir=None,
        #                                 pe_type="spd",
        #                                 pe_kwargs={"num_anchors": 3, "cutoff": 4})
        torch.save(test_in_graph, f'{graph_dir}/test_graph.pt')

    print('\nTrain')
    print(f"Mean of neighbour bound: {train_in_graph.edge_index.shape[1] // train_in_graph.x.shape[0]}")
    print('Test')
    print(f"Mean of neighbour bound: {test_in_graph.edge_index.shape[1] // test_in_graph.x.shape[0]}")

    # Create dataloaders
    train_in_graph.y = torch.from_numpy(train_labels)
    # batch_size = config['BATCH_SIZE']
    # train_loader = NeighborLoader(
    #     train_in_graph,
    #     num_neighbors=[40, 10],
    #     batch_size=batch_size,
    #     shuffle=True
    # )

    train_in_graph = train_in_graph.to('cpu')
    train_loader = RandomNodeLoader(train_in_graph, num_parts=50, shuffle=True)

    train_loader_iterator = InfiniteDataLoader(train_loader)
    miniepoch_len = 100

    test_loader = NeighborLoader(
        test_in_graph,
        num_neighbors=[-1, -1],
        batch_size=512,
        shuffle=False
    )
    train_eval_loader = NeighborLoader(
        train_in_graph,
        num_neighbors=[-1, -1],
        batch_size=512,
        shuffle=False
    )

    val_label_path = config['VAL_LABELS']
    val_labels = load_labels(val_label_path)
    val_classes = len(np.unique(val_labels))
    print(f'num of labels: {val_labels.shape[0]}')
    print(f'#cls: {val_classes}')

    # Create model
    model = GCN(in_dim=feat_dim, hidden_dim=config['HIDDEN_DIM'], out_dim=config['OUT_DIM'], dropout=config['DROPOUT'])
    print(f'GCN model:\n {model}\n')

    # Checkpoint dir
    model_dir = os.path.join(config['CHECKPOINT_DIR'], config['MODEL_NAME'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tb_writer = SummaryWriter(log_dir=model_dir) # init tb writer
    shutil.copy(config_path, model_dir)

    # Run training
    epochs = config['EPOCHS']
    lr=config['LR']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=epochs,
                                                    pct_start=0.02,
                                                    div_factor=2)
    model = model.to(device)
    best_fp = -np.Inf

    # Negatives sampler
    neg_sampler = EntropyNegativeSampler(tb_writer,
                                         neg_multiplier=config.get("neg_multiplier", 5.0),
                                         pool_factor=50,
                                         alpha=1.0,
                                         delta_start=10.0,
                                         delta_end=3.0,
                                         anneal_frac=0.8)

    for epoch in range(epochs):
        model.train()

        # Evaluation in step 0 is for pre-training metrics
        if epoch > -1:
            epoch_loss = train(model,
                               optimizer,
                               epoch,
                               epochs,
                               neg_sampler,
                               train_loader_iterator,
                               miniepoch_len,
                               config,
                               device)
            scheduler.step()

            # Add to tensorboard
            tb_writer.add_scalar('Train/Loss', epoch_loss, epoch)
            print(f'Epoch: {epoch:05d}, Loss: {epoch_loss:.4f}')

        # Evaluate
        if epoch > 40 and epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate model on the test graph, optimize tau
                print('Evaluate test graph:')
                test_x = inference(test_in_graph,
                                   model,
                                   test_loader,
                                   config,
                                   device)
                pred_labels = face_cluster(Data(test_x, test_in_graph.edge_index),
                                           no_self_links=True)
                avg_pre_p, avg_rec_p, fscore_p = pairwise(val_labels, pred_labels)
                avg_pre_b, avg_rec_b, fscore_b = bcubed(val_labels, pred_labels)

                # free up gpu memory
                torch.cuda.empty_cache()

                # Save the best checkpoint
                if fscore_p > best_fp:
                    best_fp = fscore_p
                    print(f'\nNew best epoch: {epoch}\n')
                    best_model = glob.glob(os.path.join(model_dir, 'model_best-*.pth'))
                    if len(best_model):
                        os.remove(best_model[0])
                    torch.save(model.state_dict(), os.path.join(model_dir, f'model_best-{epoch}.pth'))

                print('#pairwise: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre_p, avg_rec_p, fscore_p))
                tb_writer.add_scalar('Val_Metrics/precision_p', avg_pre_p, epoch)
                tb_writer.add_scalar('Val_Metrics/recall_p', avg_rec_p, epoch)
                tb_writer.add_scalar('Val_Metrics/Fp', fscore_p, epoch)

                print('#bicubic: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre_b, avg_rec_b, fscore_b))
                tb_writer.add_scalar('Val_Metrics/precision_b', avg_pre_b, epoch)
                tb_writer.add_scalar('Val_Metrics/recall_b', avg_rec_b, epoch)
                tb_writer.add_scalar('Val_Metrics/Fb', fscore_b, epoch)


                # Calculate train metrics
                print('Evaluate train graph:')
                train_x = inference(train_in_graph,
                                        model,
                                        train_eval_loader,
                                        config,
                                        device)

                pred_labels = face_cluster(Data(train_x, train_in_graph.edge_index),
                                           no_self_links=True)
                avg_pre_p, avg_rec_p, fscore_p = pairwise(train_labels, pred_labels)
                avg_pre_b, avg_rec_b, fscore_b = bcubed(train_labels, pred_labels)

                print('#pairwise: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre_p, avg_rec_p, fscore_p))
                tb_writer.add_scalar('Train_Metrics/precision_p', avg_pre_p, epoch)
                tb_writer.add_scalar('Train_Metrics/recall_p', avg_rec_p, epoch)
                tb_writer.add_scalar('Train_Metrics/Fp', fscore_p, epoch)

                print('#bicubic: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre_b, avg_rec_b, fscore_b))
                tb_writer.add_scalar('Train_Metrics/precision_b', avg_pre_b, epoch)
                tb_writer.add_scalar('Train_Metrics/recall_b', avg_rec_b, epoch)
                tb_writer.add_scalar('Train_Metrics/Fb', fscore_b, epoch)

                # free up gpu memory
                torch.cuda.empty_cache()

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default='configs/train.yaml',
                        type=str,
                        help='Path of the config file')
    parser.add_argument("--no-cuda",
                        action='store_true',
                        help='Do not use GPU resources')

    args = parser.parse_args()

    # Set torch device
    if (not args.no_cuda) and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main(args.config, device)