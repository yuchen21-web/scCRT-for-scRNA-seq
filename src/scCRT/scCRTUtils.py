import model.Model
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import deque

import rpy2.robjects as robjects
from rpy2.robjects import r, numpy2ri, pandas2ri, globalenv

import torch
from sklearn.metrics.pairwise import pairwise_distances
from torch import optim
from scanpy.preprocessing._deprecated.highly_variable_genes import *
import scanpy as sc

from scCRT.model.Model import *


numpy2ri.activate()
pandas2ri.activate()
robjects.r('library(magrittr)')

# 获得最小生成树

def get_mst_tree(hidden_clusters_links, num_clusters, start_node):
    tree = minimum_spanning_tree(np.max(hidden_clusters_links) + 1 - hidden_clusters_links)
    index_mapping = np.array([c for c in range(num_clusters)])
    connections = {k: list() for k in range(num_clusters)}
    cx = tree.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        i = index_mapping[i]
        j = index_mapping[j]
        connections[i].append(j)
        connections[j].append(i)
    visited = [False for _ in range(num_clusters)]
    queue = list()
    queue.append(start_node)
    children = {k: list() for k in range(num_clusters)}
    while len(queue) > 0: # BFS to construct children dict
        current_node = queue.pop()
        visited[current_node] = True
        for child in connections[current_node]:
            if not visited[child]:
                children[current_node].append(child)
                queue.append(child)
    return children


def recurse_branches(path, v, tree, branch_clusters):
    num_children = len(tree[v])
    if num_children == 0:  # at leaf, add a None token
        return path + [v, None]
    elif num_children == 1:
        return recurse_branches(path + [v], tree[v][0], tree, branch_clusters)
    else:  # at branch
        branch_clusters.append(v)
        return [recurse_branches(path + [v], tree[v][i], tree, branch_clusters) for i in range(num_children)]

def flatten(li):
    if li[-1] is None:  # special None token indicates a leaf
        yield li[:-1]
    else:  # otherwise yield from children
        for l in li:
            yield from flatten(l)


class Lineage:
    class SingleLineage:
        def __init__(self, lineage, lineage_seq, lineage_index):
            self.lineage = lineage  # 轨迹包含的clusters的顺序(完整)
            self.lineage_seq = lineage_seq  # 单个轨迹相邻cluster的顺序(相邻)
            self.lineage_index = lineage_index  # 该轨迹在 所有轨迹 lineage_seq 的index [a, b] -> 0

        def __repr__(self):
            return 'lineage:' + str(self.lineage) + ', ' + \
                   'lineage_seq:' + str(self.lineage_seq) + ', ' + \
                   'lineage_index:' + str(self.lineage_index) + '\n'

    def __init__(self):
        '''
        all: [[a, b], [b, c], [c, d], [c, f]] 所有轨迹相邻cluster的顺序
        self class:
            self: [a, b, c, f] 单个轨迹的顺序
            self: [[a, b], [b, c], [c, f]] 单个轨迹相邻cluster的顺序
            self: [0, 1, 3] 单个轨迹相邻cluster的顺序在所有轨迹中的index

        all cluster: [[a, b, c], [f, d, e], [a, d, e]] --> a:[0, 2]  # 类属于单个整条轨迹在 轨迹list的所属下标
        all cluster: [cluster, all_lineage_seq]: [a, b, c] -> [[a, b], [b, c]] -> [0, 1] -> mask矩阵中a类的 0,1 置1
        '''
        self.all_lineage_seq = None  # 所有轨迹，两两cluster的前后顺序 [a, b, c] -> [[a, b], [b, c]]
        self.lineages = []  # 包含每条轨迹的list

        self.cluster_lineages = None  # 每个类属于单个轨迹(完整路径)，指的是被 lineage list 所包含下标
        self.cluster_lineseq_mat = None  # 每个类的所属轨迹(完整路径)的两相邻类在all_lineage_seq所属下标的mask

    def add_lineage(self, lineage, lineage_seq, lineage_index):
        self.lineages.append(self.SingleLineage(lineage, lineage_seq, lineage_index))

    def __repr__(self):
        return 'all_lineage_seq: ' + \
               str(self.all_lineage_seq.tolist()) + \
               '\nlineages: ' + '\n' + str(self.lineages)

def getInnerClusterKNN(cell_labels, features, k=5, norm=2):
    # 用于查看降维后各个cluster之间的连接状况


    adj_hidden_dist = pairwise_distances(features, metric='euclidean')
    # adj_hidden_dist = pairwise_distances(pca_features, metric='euclidean')
    # 消除内部连接
    max_dis = np.max(adj_hidden_dist)
    for i in range(adj_hidden_dist.shape[0]):
        cell_type = cell_labels[i]
        adj_hidden_dist[i][cell_labels == cell_type] = max_dis

    # argsort 是从小到大
    sorted_near_neighbour_hidden = adj_hidden_dist.argsort(axis=-1)
    cell_nums = sorted_near_neighbour_hidden.shape[0]
    adj_hidden_dense = np.zeros([cell_nums, cell_nums])
    for i in np.arange(cell_nums):
        for j in range(k):
            # cosine 是 -j
            # euclid 是 j
            adj_hidden_dense[i, sorted_near_neighbour_hidden[i][(j + 1)]] = 1 - 0.02 * j  # -j从最大处开始选择
    #         adj_hidden_dense[i, sorted_near_neighbour_hidden[i][(j+1)]] = 1
    #         adj_hidden_dense[sorted_near_neighbour_hidden[i][-j], i] = 1

    labels = cell_labels
    # labels = partition_label
    label_sets = set(labels)
    cell_nums_dict = {}
    for i in set(labels):
        cell_nums_dict[i] = sum(labels == i)
    hidden_clusters_links = np.zeros([len(label_sets), len(label_sets)], dtype=float)
    hidden_clusters_links_norm = np.zeros([len(label_sets), len(label_sets)], dtype=float)
    for i in range(len(label_sets)):
        for j in range(len(label_sets)):
            if i == j:
                continue
            if i > j:
                # 这里不直接用 hidden_clusters_links[i, j]=[j, i] 的原因是构建adj_hidden_dense就没有对称
                hidden_clusters_links[i, j] = np.sum(adj_hidden_dense[labels == i].T[labels == j])
                hidden_clusters_links_norm[i, j] = hidden_clusters_links[i, j] / np.power(np.abs(
                    cell_nums_dict[j] - cell_nums_dict[i]) + 1, 1 / norm)
                continue

            hidden_clusters_links[i, j] = np.sum(adj_hidden_dense[labels == i].T[labels == j])
            hidden_clusters_links_norm[i, j] = hidden_clusters_links[i, j] / np.power(np.abs(
                cell_nums_dict[j] - cell_nums_dict[i]) + 1, 1 / norm)

    return hidden_clusters_links, hidden_clusters_links_norm


def getLineage(links, cell_labels, start_node=0):
    '''
    使用学习到的特征, 类内连接度计算 簇

    branch_clusters (list): # 产生分支的节点 cluster
    lineages_list (list): 轨迹多个路径的list
    cluster_lineages (dict): 每个cluster所属于lineages的路径序号

    Lineage class:
        - lineage (SingleLineage class list):
                - lineage: 单个轨迹的顺序 [a, b, c]
                - lineage_seq: 单个轨迹相邻类的顺序 [[a, b], [b, c]]
                - lineage_index: 单个轨迹相邻类的顺序在所有顺序all_lineage_seq的下标 [0, 3]
        - all_lineage_seq (list): 所有轨迹的相邻类的顺序 [[a, b], [b, e], [b, c]]
        - cluster_lineages (dict): 每个cluster所属于轨迹在轨迹list lineage的下标
        - cluster_lineseq_mat (mask): 每个cluster所属于 所有轨迹的lineage_seq 在lineage_index的下标 位置置1

    '''
    ########### 计算得到 lineages_list
    num_clusters = len(set(cell_labels))
    tree = get_mst_tree(links, num_clusters, start_node=start_node)
    branch_clusters = deque()  # 产生分支的节点 cluster
    lineages_list = recurse_branches([], start_node, tree, branch_clusters)
    lineages_list = list(flatten(lineages_list))

    ########### 构造 lineages class Infered_lineages
    # 包含 、单个轨迹中相邻cluster的顺序lineage_seq list
    Infered_lineages = Lineage()
    ## lineage_seq: 得到lineages后，用于存储一条轨迹的相邻cluster [a, b, c] -> [[a, b], [b, c]]
    lineage_seq = []
    # adj_cluster2lineage_seq: 一条轨迹的相邻cluster在lineage_seq的下标, 例如 [[a, b], [b, c]] ->  [a, b] -> 0
    adj_cluster2lineage_seq = dict()

    for each_line in lineages_list:
        for i in range(len(each_line) - 1):
            str_seq = str(each_line[i]) + str(each_line[i + 1])
            if str_seq in adj_cluster2lineage_seq.keys():
                continue
            adj_cluster2lineage_seq[str_seq] = len(lineage_seq)  # 存储lineage_seq的index
            lineage_seq.append([each_line[i], each_line[i + 1]])
    lineage_seq = np.array(lineage_seq)
    Infered_lineages.all_lineage_seq = lineage_seq

    # 该cluster属于lineages的几个分支
    cluster_lineages = {k: list() for k in range(num_clusters)}
    for l_idx, lineage in enumerate(lineages_list):
        for k in lineage:
            cluster_lineages[k].append(l_idx)
    Infered_lineages.cluster_lineages = cluster_lineages

    ## 该cluster的所属轨迹(每相邻两个cluster center)，属于lineage_seq的index会为1
    # cluster_lineseq_mat: 一个mask矩阵 [cluster_num, lineage_seq_num];
    cluster_lineseq_mat = np.zeros([len(set(cell_labels)), len(lineage_seq)])
    for each_line in lineages_list:
        each_line_index = []  # 存储这条谱系中出现的相邻线段在lineage_seq的index
        each_line_seq = []
        for i in range(len(each_line) - 1):
            each_line_seq.append([each_line[i], each_line[i + 1]])
            str_seq = str(each_line[i]) + str(each_line[i + 1])
            each_line_index.append(adj_cluster2lineage_seq[str_seq])

        each_line_index = np.array(each_line_index)

        # 添加 lineage class 的顺序
        Infered_lineages.add_lineage(each_line, each_line_seq, each_line_index)  # 要加相邻的东西

        for i in each_line:
            cluster_lineseq_mat[i, each_line_index] = 1
    Infered_lineages.cluster_lineseq_mat = cluster_lineseq_mat

    return Infered_lineages, branch_clusters


def get_cell_lineages_info(Infered_lineages, features, cell_labels, sep_points=99):
    ######################################### 用于得出dynplot画图所需的数据 #########################################

    # 为轨迹分割，并将每个cell赋予至轨迹

    # 分割的比例

    # 用于求出 dynplot 绘图所需的数据
    line_points = []  # 下标为index的point在这条轨迹上的位置
    line_points_percent = []  # 下标为index的point在这条轨迹上的位置比例
    line_points_Type = []  # 下标为index的point所属于 Infered_lineages.lineage 的第几条轨迹

    centers = []
    for label_id in range(len(set(cell_labels))):
        centers.append(np.mean(features[cell_labels == label_id], axis=0))
    centers = np.array(centers)

    lineage_seq = Infered_lineages.all_lineage_seq

    for index, (i, j) in enumerate(lineage_seq):
        points_x = centers[i]
        points_y = centers[j]
        dis_xy = (points_y - points_x) / sep_points
        for i in range(sep_points):
            #         each_line_nodes.append(points_x + i*dis_xy)
            line_points.append(points_x + i * dis_xy)  # 加入当前point的位置
            line_points_percent.append(i / (sep_points + 1))  # 加入当前point的比例
            line_points_Type.append(index)  # 加入当前在第几条轨迹上
        #     each_line_nodes.append(points_y)
        line_points.append(points_y)
        line_points_percent.append(1)
        line_points_Type.append(index)
    #     line_nodes.append(each_line_nodes)

    line_points = np.array(line_points)
    line_points_percent = np.array(line_points_percent)
    line_points_Type = np.array(line_points_Type)


    cell_point_dist = pairwise_distances(features,
                                         line_points,
                                         metric='euclidean')

    # 因为原属于line上的cluster可能会分到别的line， 所以先将不属于该cluster的line调大
    # 思想： 将不属于cluster的line，例如有 [3, 0, 1, 7]，那么7必定不能分配到[0, 5],或者[1, 4]上
    tmax = np.max(cell_point_dist)
    for i in set(cell_labels):
        filter_types = (1 - Infered_lineages.cluster_lineseq_mat[i]).nonzero()[0]
        if len(filter_types) > 0:
            for f_type in filter_types:
                #             cell_point_dist[cell_labels==i, :][:, line_points_Type==f_type] = 0 # 这个无效
                for index in (cell_labels == i).nonzero()[0]:
                    cell_point_dist[index, line_points_Type == f_type] = tmax

    cell_belong_point = cell_point_dist.argsort()[:, 0]

    return line_points_percent, line_points_Type, cell_belong_point


def getInputData(dataset_path, dataset_label_path=None):
    # rds format need use R
    if '.rds' in dataset_path:
        return getInfoFromR(dataset_path)
    elif '.csv' in dataset_path:
        return getInfoFromCsv(dataset_path, dataset_label_path)


def getInfoFromCsv(dataset_path, dataset_label_path=None):
    data_csv = pd.read_csv(dataset_path, index_col=0)
    genes = data_csv.columns.values
    cells = data_csv.index.values
    data_counts = data_csv.values

    if dataset_label_path is not None:
        label_csv = pd.read_csv(dataset_label_path, index_col=0)
        cell_labels_name = label_csv.values.T[1]
        name2ids = {}
        ids2name = {}
        for cell_names in np.sort(list(set(cell_labels_name))):
            name2ids[cell_names] = len(name2ids)
            ids2name[name2ids[cell_names]] = cell_names

        cell_labels = np.array([name2ids[i] for i in cell_labels_name])

        if label_csv.shape[-1] == 3:
            cell_times = label_csv.values.T[2].astype(float)
        else:
            cell_times = None

        return data_counts, cell_labels, cells, name2ids, ids2name, cell_times, None
    return data_counts, None, cells, None, None, None, None


def getInfoFromR(dataset_path):
    globalenv['dataset_path'] = dataset_path
    pre_infos = r("""
    data <- readRDS(dataset_path)

    groups_id <- data$prior_information$groups_id
    labels <- rownames(data$counts) %>% as.data.frame()%>% 
      plyr::rename(replace = c('.'='cell_id')) %>%
      dplyr::left_join(groups_id, by='cell_id')

    labels <- data[["prior_information"]][["timecourse_continuous"]] %>% 
      as.data.frame() %>% plyr::rename(replace = c('.'='real_time')) %>% 
      tibble::rownames_to_column('cell_id') %>%
      dplyr::left_join(labels, ., by='cell_id')

    start_cluster_id <- data$prior_information$start_milestones

    list(data=data, labels=labels, start_id=start_cluster_id)
    """)

    globalenv['pre_infos'] = pre_infos

    ## 数据矩阵
    data_counts = r("""
    pre_infos$data$counts
    """).astype(int)

    ## labels
    cell_infos = pandas2ri.rpy2py(pre_infos[1]).values.T

    cells = cell_infos[0]
    cell_labels_name = cell_infos[1]
    cell_times = cell_infos[2]

    name2ids = {}
    ids2name = {}
    for cell_names in np.sort(list(set(cell_labels_name))):
        name2ids[cell_names] = len(name2ids)
        ids2name[name2ids[cell_names]] = cell_names

    cell_labels = np.array([name2ids[i] for i in cell_labels_name])

    return data_counts, cell_labels, cells, name2ids, ids2name, cell_times, pre_infos


def pre_process(data_counts, WithoutLabel=False):
    # preprocess
    # 数据预处理，一次性, 输入AnnData: [cells, genes]
    adata = sc.AnnData(data_counts, dtype=int)
    sc.pp.normalize_total(adata, key_added='n_counts_all')

    n_top_genes = data_counts.shape[-1]
    set_topGene = 2000
    if n_top_genes > set_topGene:
        filter_result = filter_genes_dispersion(
            adata.X, flavor='cell_ranger', n_top_genes=set_topGene, log=False
        )

        adata = sc.AnnData(data_counts)

        adata._inplace_subset_var(filter_result.gene_subset)
        sc.pp.normalize_total(adata)

    sc.pp.log1p(adata)
    features = np.copy(adata.X)

    if WithoutLabel:
        sc.pp.scale(adata)
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
        sc.tl.louvain(adata, resolution=0.2)

        cell_labels = adata.obs['louvain'].astype(int).tolist()

        return features, cell_labels

    return features


def trainingModel(features, cell_labels, device, hidden=[128, 32], k=20, epochs=200, model_path=None):
    if model_path is not None:
        X = torch.FloatTensor(features).to(device)
        model = scCRT(X.shape[-1], hidden[0], hidden[1])
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        y, _ = model(X)
        return y.detach().cpu().numpy()

    adj_dist = pairwise_distances(features, metric='euclidean')  # manhattan
    sorted_near_neighbour = adj_dist.argsort(axis=-1)  # 排序后的邻居

    positive_cell = sorted_near_neighbour[:, 1:k + 1]
    negative_cell = sorted_near_neighbour[:, -int(sorted_near_neighbour.shape[0] / 5):]

    X = torch.FloatTensor(features).to(device)

    model = scCRT(X.shape[-1], hidden[0], hidden[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_pair = Cell_pair_loss()
    criterion_cluster = Cluster_cons_loss(cell_labels, device)

    for i in range(epochs):
        model.train()
        optimizer.zero_grad()

        pos_cell, neg_cell = getPosNegData(positive_cell, negative_cell)

        y, _ = model(X)

        pair_loss = criterion_pair(y, pos_cell, neg_cell)
        center_loss = criterion_cluster(y, rate=0.3)

        loss = pair_loss + center_loss

        loss.backward()
        optimizer.step()

    # torch.save(model.cpu().state_dict(), 'binary_tree_8_model.pkl')

    return y.detach().cpu().numpy()


def get_trajectory(cell_labels, y_features, ids2name, cells, start_node=0, norm=False, k=20):
    #     start_cluster = r("""
    #     pre_infos$start_id %>% as.data.frame()
    #     """).values[0].squeeze()

    clusters_links, clusters_links_norm = getInnerClusterKNN(cell_labels, y_features, k=k, norm=2)  # 计算类间连接
    if norm:
        Lineage_class, _ = getLineage(clusters_links_norm,
                                      cell_labels, start_node=start_node)  # 得到轨迹
    else:
        Lineage_class, _ = getLineage(clusters_links, cell_labels, start_node=start_node)  # 得到轨迹
    points_percent, points_Type, cell_belong_point = get_cell_lineages_info(Lineage_class, y_features,
                                                                            cell_labels, sep_points=199)
    cell_belong_type = points_Type[cell_belong_point]

    ###########################   dynplot 的数据结构 ####################################
    #### progressions
    ## percentage
    percentage = points_percent[cell_belong_point].astype(float)

    lineage_seq = Lineage_class.all_lineage_seq
    ## from, to
    # cell_line_type: lineage上每个point所属的type
    # line_points_Type: 所有cell所属的type
    cell_line_type = points_Type[cell_belong_point]  # 每个point所属的type (lineage_seq 的 index)
    start2target = lineage_seq[cell_line_type].T  # 从lineage_seq取出每个point的 start和target

    if ids2name is not None:
        cell_starts = [ids2name[i] for i in start2target[0]]
        cell_targets = [ids2name[i] for i in start2target[1]]

        network_starts = [ids2name[i] for i in lineage_seq.T[0]]
        network_targets = [ids2name[i] for i in lineage_seq.T[1]]
    else:
        cell_starts = [i for i in start2target[0]]
        cell_targets = [i for i in start2target[1]]

        network_starts = [i for i in lineage_seq.T[0]]
        network_targets = [i for i in lineage_seq.T[1]]

    network_length = [np.sum(cell_line_type == i) for i in range(len(lineage_seq))]
    network_length = network_length / max(network_length) * 2
    network_directed = np.ones(len(network_starts)).astype(bool)

    progressions = pd.DataFrame(np.array([cells, cell_starts, cell_targets, percentage]).astype(str).T,
                                index=None, columns=['cell_id', 'from', 'to', 'percentage'])

    milestone_percentages = pd.DataFrame(np.array([cells, cell_starts, np.zeros(cells.shape)]).astype(str).T,
                                         index=None, columns=['cell_id', 'milestone_id', 'percentage'])

    milestone_network = pd.DataFrame(np.array([network_starts, network_targets, network_length, network_directed]).T,
                                     index=None, columns=['from', 'to', 'length', 'directed'])

    return Lineage_class, [network_starts, network_targets], [progressions, milestone_percentages, milestone_network]


def evaluation(pre_infos, evaluation_details):
    progressions, milestone_percentages, milestone_network = evaluation_details
    globalenv['progressions'] = pre_infos
    globalenv['progressions'] = pandas2ri.py2rpy(progressions)
    globalenv['milestone_percentages'] = pandas2ri.py2rpy(milestone_percentages)
    globalenv['milestone_network'] = pandas2ri.py2rpy(milestone_network)

    info4metric = r("""

    progressions <- tibble::as_tibble(progressions)
    progressions$percentage <- as.numeric(progressions$percentage)

    milestone_percentages <- tibble::as_tibble(milestone_percentages)
    milestone_percentages$percentage <- as.numeric(milestone_percentages$percentage)

    milestone_network <- tibble::as_tibble(milestone_network)
    milestone_network$directed <- as.logical(milestone_network$directed)
    milestone_network$length <- as.numeric(milestone_network$length)
    milestone_network$length = 1

    info4metric <- dynwrap::wrap_expression(
      counts = pre_infos$data$counts,
      expression = pre_infos$data$expression
      ) %>%
      dynwrap::add_trajectory(
      milestone_network = milestone_network,
      progressions = progressions,
      milestone_percentages = milestone_percentages)
    """)

    globalenv['info4metric'] = info4metric

    Him = r("""
        pre_infos$data$milestone_network$length = 1
        dyneval::calculate_metrics(pre_infos$data, info4metric, 'him')
    """).values.squeeze()[0]

    F1_branches = r("""
        dyneval::calculate_metrics(pre_infos$data, info4metric, 'F1_branches')
    """).values.squeeze()[-1]

    F1_milestones = r("""
        dyneval::calculate_metrics(pre_infos$data, info4metric, 'F1_milestones')
    """).values.squeeze()[-1]

    return [Him, F1_branches, F1_milestones]

