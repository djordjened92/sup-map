import os
import yaml
import glob
import torch
import shutil
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import GCN, map_eq_loss
from utils import *
from dataset import load_feat, load_labels, FeatureDataset
from metrics import pairwise, bcubed
from torch_scatter import scatter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pytorch_metric_learning import samplers
from torch_geometric.loader import NeighborLoader

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

def train(model, optimizer, training_loader, labels, iterations, config, device):
    running_loss = 0.

    for i in tqdm(range(iterations), 'Train loader:'):
        b_features, b_labels = next(training_loader)
        b_features = b_features.to(device)
        b_labels = b_labels.to(device)
        
        optimizer.zero_grad()  # Clear gradients.
        K = config['K']
        data, nbrs = gen_graph(b_features, config, K, device)
        out = model(data.x, data.edge_index)  # Perform a single forward pass.

        # init_codelength = map_eq_loss(data.x, nbrs, b_labels)
        # codelength = map_eq_loss(out, nbrs, b_labels)
        # curr_loss = torch.clamp(codelength - init_codelength, min=init_codelength)
        curr_loss = map_eq_loss(out, nbrs, b_labels)

        curr_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        running_loss += curr_loss.item()

        # free up gpu memory
        torch.cuda.empty_cache()

    return running_loss / iterations

def inference(in_graph, in_nbrs, in_ji, model, data_loader, config, device):
    out = torch.zeros((in_graph.num_nodes, config['OUT_DIM']), device=device)
    for data in data_loader:
        local_idx = (data.input_id[:, None]==data.n_id).nonzero()[:, 1]
        out[data.input_id, :] = model(data.x, data.edge_index)[local_idx]
    torch.cuda.empty_cache()

    # Calculate similarities for the same neighbours before GCN inference
    # We calculate row-by-row similarities due to VRAM limitation
    out_sims = torch.zeros_like(in_nbrs, dtype=torch.float32, device=device)
    for i, out_feat in enumerate(out):
        out_sims[i, :] = torch.clamp(out[in_nbrs[i]]@out_feat, 0.)

    return out_sims

def main(config_path, device):
    # Load config
    config = yaml.safe_load(open(config_path, 'r'))
    k = config['K']

    # Load data
    feat_dim = config['FEATURE_DIM']

    train_feat_path = config['TRAIN_FEATURES']
    train_feat = load_feat(train_feat_path, feat_dim)
    train_features = l2norm(train_feat)
    print('Train:')
    print('features shape:', train_features.shape)

    train_label_path = config['TRAIN_LABELS']
    train_labels = load_labels(train_label_path)
    print(f'num of labels: {train_labels.shape[0]}')
    print(f'#cls: {len(np.unique(train_labels))}\n')

    graph_dir = f'ws{config["WINDOW_SIZE"]}'

    if os.path.exists(graph_dir):
        test_in_graph = torch.load(f'{graph_dir}/test_graph.pt', weights_only=False)
        test_nbrs_bounds = torch.load(f'{graph_dir}/test_nbrs_bounds.pt', weights_only=False)
        test_in_graph['nbrs_bounds'] = test_nbrs_bounds
        test_nbrs = torch.load(f'{graph_dir}/test_nbrs.pt', weights_only=False)
        test_ji = torch.load(f'{graph_dir}/test_ji.pt', weights_only=False)

        train_in_graph = torch.load(f'{graph_dir}/train_graph.pt', weights_only=False)
        train_nbrs_bounds = torch.load(f'{graph_dir}/train_nbrs_bounds.pt', weights_only=False)
        train_in_graph['nbrs_bounds'] = train_nbrs_bounds
        train_nbrs = torch.load(f'{graph_dir}/train_nbrs.pt', weights_only=False)
        train_ji = torch.load(f'{graph_dir}/train_ji.pt', weights_only=False)
    else:
        os.makedirs(graph_dir)

        # Generate train graph
        train_feat_path = config['TRAIN_FEATURES']
        train_feat = load_feat(train_feat_path, feat_dim)
        train_features = l2norm(train_feat)
        print('Train:')
        print('features shape:', train_features.shape)
        train_in_graph, train_nbrs = gen_graph(torch.from_numpy(train_features).to(device),
                                                        config,
                                                        80,
                                                        device,
                                                        z_score=True)
        train_ji = jaccard_index(train_nbrs,
                                torch.zeros_like(train_nbrs, device=device).float(),
                                1.,
                                train_in_graph['nbrs_bounds'])
        torch.save(train_in_graph, f'{graph_dir}/train_graph.pt')
        torch.save(train_in_graph['nbrs_bounds'], f'{graph_dir}/train_nbrs_bounds.pt')
        torch.save(train_nbrs, f'{graph_dir}/train_nbrs.pt')
        torch.save(train_ji, f'{graph_dir}/train_ji.pt')

        # Generate test graph 
        val_feat_path = config['VAL_FEATURES']
        val_feat = load_feat(val_feat_path, feat_dim)
        val_features = l2norm(val_feat)
        print('Test:')
        print('features shape:', val_features.shape)
        test_in_graph, test_nbrs = gen_graph(torch.from_numpy(val_features).to(device),
                                                        config,
                                                        80,
                                                        device,
                                                        z_score=True)
        test_ji = jaccard_index(test_nbrs,
                                torch.zeros_like(test_nbrs, device=device).float(),
                                1.,
                                test_in_graph['nbrs_bounds'])
        torch.save(test_in_graph, f'{graph_dir}/test_graph.pt')
        torch.save(test_in_graph['nbrs_bounds'], f'{graph_dir}/test_nbrs_bounds.pt')
        torch.save(test_nbrs, f'{graph_dir}/test_nbrs.pt')
        torch.save(test_ji, f'{graph_dir}/test_ji.pt')

    print('\nTrain')
    train_K_median = train_in_graph['nbrs_bounds'].median().int()
    print(f"Median of neighbour bound: {train_K_median}")
    print(f"Mean of neighbour bound: {train_in_graph['nbrs_bounds'].float().mean()}")
    print('Test')
    test_K_median = test_in_graph['nbrs_bounds'].median().int()
    print(f"Median of neighbour bound: {test_K_median}")
    print(f"Mean of neighbour bound: {test_in_graph['nbrs_bounds'].float().mean()}")

    batch_size = config['BATCH_SIZE']
    # train_loader = NeighborLoader(
    #     train_in_graph,
    #     num_neighbors=[-1, -1],
    #     batch_size=batch_size,
    #     shuffle=True
    # )
    test_loader = NeighborLoader(
        test_in_graph,
        num_neighbors=[-1, -1],
        batch_size=512,
        shuffle=False
    )

    val_label_path = config['VAL_LABELS']
    val_labels = load_labels(val_label_path)
    val_classes = len(np.unique(val_labels))
    print(f'num of labels: {val_labels.shape[0]}')
    print(f'#cls: {val_classes}')

    # Create Dataset
    train_dataset = FeatureDataset(train_features, train_labels)
    sampler = samplers.MPerClassSampler(labels=train_labels,
                                        m=config['SAMPLES_PER_CLASS'],
                                        batch_size=batch_size,
                                        length_before_new_iter=250000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    # Create model
    model = GCN(in_dim=feat_dim, hidden_dim=config['HIDDEN_DIM'], out_dim=config['OUT_DIM'], dropout=config['DROPOUT'])
    print(f'GCN model:\n {model}\n')

    # ckpt_dir = '/home/djordje/Documents/Projects/face-mod/checkpoints/bs60_k60_lr1.1e-4_ep500_do0.15_012/model_best-232.pth'
    # state_dict = torch.load(ckpt_dir)
    # model.load_state_dict(state_dict)

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

    # for _ in range(232):
    #     scheduler.step()
    # Initial cluster metrics
    # test_sims = torch.zeros_like(test_nbrs, dtype=torch.float32, device=device)
    # for i, out_feat in enumerate(test_in_graph.x):
    #     test_sims[i, :] = torch.clamp(test_in_graph.x[test_nbrs[i]]@out_feat, 0.)
    # torch.cuda.empty_cache()
    # pred_labels = face_cluster(test_nbrs.cpu().numpy(), test_sims.cpu().numpy())
    # avg_pre_p, avg_rec_p, fscore_p = pairwise(val_labels, pred_labels)
    # avg_pre_b, avg_rec_b, fscore_b = bcubed(val_labels, pred_labels)
    # print('\nInitial metrics')
    # print('#pairwise: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre_p, avg_rec_p, fscore_p))
    # print('#bicubic: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}\n'.format(avg_pre_b, avg_rec_b, fscore_b))

    train_loader_iterator = InfiniteDataLoader(train_loader)
    miniepoch_len = 200
    for epoch in range(1, epochs):
        model.train()
        epoch_loss = train(model,
                           optimizer,
                           train_loader_iterator,
                           torch.from_numpy(train_labels),
                           miniepoch_len,
                           config,
                           device)
        scheduler.step()

        # Add to tensorboard
        tb_writer.add_scalar('Train/Loss', epoch_loss, epoch)

        # Evaluate
        if epoch > 0 and epoch % 8 == 0:
            print(f'Epoch: {epoch:05d}, Loss: {epoch_loss:.4f}')

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
                # print('Evaluate train graph:')
                # train_sims = inference(train_in_graph,
                #                         train_nbrs,
                #                         train_ji,
                #                         model,
                #                         train_loader,
                #                         config,
                #                         device)

                # pred_labels = face_cluster(train_nbrs.cpu().numpy(), train_sims.cpu().numpy())
                # avg_pre_p, avg_rec_p, fscore_p = pairwise(train_labels, pred_labels)
                # avg_pre_b, avg_rec_b, fscore_b = bcubed(train_labels, pred_labels)

                # print('#pairwise: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre_p, avg_rec_p, fscore_p))
                # tb_writer.add_scalar('Train_Metrics/precision_p', avg_pre_p, epoch)
                # tb_writer.add_scalar('Train_Metrics/recall_p', avg_rec_p, epoch)
                # tb_writer.add_scalar('Train_Metrics/Fp', fscore_p, epoch)

                # print('#bicubic: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre_b, avg_rec_b, fscore_b))
                # tb_writer.add_scalar('Train_Metrics/precision_b', avg_pre_b, epoch)
                # tb_writer.add_scalar('Train_Metrics/recall_b', avg_rec_b, epoch)
                # tb_writer.add_scalar('Train_Metrics/Fb', fscore_b, epoch)

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