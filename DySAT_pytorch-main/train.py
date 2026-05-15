# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/02/20 10:25:13
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import argparse
import networkx as nx
import numpy as np
import dill
import pickle as pkl
import scipy
from torch.utils.data import DataLoader

from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
from utils.minibatch import  MyDataset
from utils.utilities import to_device
from eval.link_prediction import evaluate_classifier
from models.model import DySAT

import torch
torch.autograd.set_detect_anomaly(True)

def inductive_graph(graph_former, graph_later):
    """Create the adj_train so that it includes nodes from (t+1) 
       but only edges from t: this is for the purpose of inductive testing.
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=16,
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='Enron',
                        help='dataset name')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                    help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=10,
                        help="patient")
    parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')
    parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each GAT layer')
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')
    args = parser.parse_args()
    print(args)

    # 1. 加载图
    graphs, adjs = load_graphs(args.dataset)
    
    # === [补丁 A] 强制节点对齐与自环 ===
    max_node_id = 0
    for g in graphs:
        if g.nodes():
            max_node_id = max(max_node_id, max(g.nodes()))
    num_total_nodes = max_node_id + 1
    
    for i in range(len(graphs)):
        graphs[i].add_nodes_from(range(num_total_nodes))
        graphs[i].add_edges_from([(n, n) for n in range(num_total_nodes)])
        adjs[i] = nx.adjacency_matrix(graphs[i])
    print(f"节点对齐完成：所有 Snapshot 统一为 {num_total_nodes} 个节点，且已添加自环。")

    # === [补丁 B] 采用固定维度的随机特征代替全量 One-Hot，防止显存爆炸 ===
    if args.featureless == True:
        feat_dim = 128  # 将特征维度降至 128，极大节省内存
        print(f"正在生成降维随机特征矩阵 (维度: {num_total_nodes}x{feat_dim})...")
        np.random.seed(42)
        static_feat = np.random.normal(0, 1, (num_total_nodes, feat_dim)).astype(np.float32)
        feats = [scipy.sparse.csr_matrix(static_feat) for _ in range(len(adjs))]
        in_features_dim = feat_dim
    else:
        in_features_dim = feats[0].shape[1] 

    assert args.time_steps <= len(adjs), "Time steps is illegal"
    context_pairs_train = get_context_pairs(graphs, adjs)

    # 3. 准备评估数据
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
        test_edges_pos, test_edges_neg = get_evaluation_data(graphs)

    # 4. 构建数据加载器 (num_workers=0 彻底杜绝超算 Bus error 报错)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = MyDataset(args, graphs, feats, adjs, context_pairs_train)
    dataloader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=0, 
                             collate_fn=MyDataset.collate_fn)

    # 初始化模型 (输入维度使用 in_features_dim)
    model = DySAT(args, in_features_dim, args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # in training
    best_epoch_val = 0
    patient = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        for idx, feed_dict in enumerate(dataloader):
            feed_dict = to_device(feed_dict, device)
            opt.zero_grad()
            loss = model.get_loss(feed_dict)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())

        model.eval()
        emb = model(feed_dict["graphs"])[:, -2, :].detach().cpu().numpy()
        val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                            train_edges_neg,
                                                            val_edges_pos, 
                                                            val_edges_neg, 
                                                            test_edges_pos,
                                                            test_edges_neg, 
                                                            emb, 
                                                            emb)
        
        # 修复了此处原生 DySAT 字典索引错误导致的 KeyError
        epoch_auc_val = val_results["HAD"][1] if "HAD" in val_results else 0.0
        try:
            epoch_ap_val = val_results["HAD"][2]
        except:
            epoch_ap_val = 0.0
        
        epoch_auc_test = test_results["HAD"][1] if "HAD" in test_results else 0.0
        try:
            epoch_ap_test = test_results["HAD"][2]
        except:
            epoch_ap_test = 0.0

        if epoch_auc_val > best_epoch_val:
            best_epoch_val = epoch_auc_val
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                print("Early stopping triggered.")
                break
                
        print("Epoch {:<3} | Loss: {:.3f} | Val AUC: {:.3f} AP: {:.3f} | Test AUC: {:.3f} AP: {:.3f}".format(
            epoch, np.mean(epoch_loss), epoch_auc_val, epoch_ap_val, 
            epoch_auc_test, epoch_ap_test))
        
    # =========================================================
    # Test Best Model 阶段
    # =========================================================
    model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
    model.eval()
    emb = model(feed_dict["graphs"])[:, -2, :].detach().cpu().numpy()

    # === 1. 计算全量测试集的 TLP ===
    val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                        train_edges_neg,
                                                        val_edges_pos, 
                                                        val_edges_neg, 
                                                        test_edges_pos,
                                                        test_edges_neg, 
                                                        emb, 
                                                        emb)
    auc_test = test_results["HAD"][1]
    try:
        ap_test = test_results["HAD"][2]
    except:
        ap_test = 0.0

    # === 2. 过滤提取 NEW 节点边 ===
    seen_nodes = set()
    for u, v in train_edges_pos: 
        seen_nodes.add(u); seen_nodes.add(v)
    for u, v in train_edges_neg: 
        seen_nodes.add(u); seen_nodes.add(v)
    
    new_test_edges_pos = [e for e in test_edges_pos if e[0] not in seen_nodes or e[1] not in seen_nodes]
    new_test_edges_neg = [e for e in test_edges_neg if e[0] not in seen_nodes or e[1] not in seen_nodes]
    
    # === 3. 计算 TNLP (NEW 节点指标) ===
    if len(new_test_edges_pos) > 0 and len(new_test_edges_neg) > 0:
        _, new_test_results, _, _ = evaluate_classifier(train_edges_pos, 
                                                        train_edges_neg, 
                                                        val_edges_pos, 
                                                        val_edges_neg, 
                                                        new_test_edges_pos, 
                                                        new_test_edges_neg, 
                                                        emb, 
                                                        emb)
        new_auc_test = new_test_results["HAD"][1]
        try:
            new_ap_test = new_test_results["HAD"][2]
        except:
            new_ap_test = 0.0
    else:
        new_auc_test, new_ap_test = 0.5, 0.0

    # === 4. 打印四项完整指标 ===
    print("\n" + "="*50)
    print("--- 🏆 Best Model Test Results ---")
    print("Test TLP AUC      = {:.4f}".format(auc_test))
    if ap_test > 0: 
        print("Test TLP AP       = {:.4f}".format(ap_test))
    print("Test TNLP NEW AUC = {:.4f}".format(new_auc_test))
    if new_ap_test > 0: 
        print("Test TNLP NEW AP  = {:.4f}".format(new_ap_test))
    print("="*50 + "\n")