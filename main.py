import time
import argparse
import pickle
import os
from model import *
from utils import *

def load_image_global_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)

opt = parser.parse_args()


def main():
    init_seed(2020)

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
    elif opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    elif opt.dataset == 'Amazon_grocery_2018':
        num_node = 11857
        opt.n_iter = 1
        opt.dropout_gcn = 0.2     # recommended values (you can adjust later)
        opt.dropout_local = 0.0
    else:
        num_node = 310

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    if opt.dataset == 'Amazon_grocery_2018':
        global_graph_path = os.path.join('datasets', opt.dataset, 'image_global_graph.pkl')
        image_graph = load_image_global_graph(global_graph_path)
        edge_index = image_graph['edge_index']
        x = image_graph['x']
    else:
        edge_index, x = None, None


    # adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    # num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)

    # adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    # model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    print("🔍 Checking max item ID in train and test:")
    max_train_id = max([max(seq) for seq in train_data.inputs if len(seq) > 0])
    max_test_id = max([max(seq) for seq in test_data.inputs if len(seq) > 0])
    print(f"Max item ID in train: {max_train_id}")
    print(f"Max item ID in test: {max_test_id}")
    print(f"num_node (embedding size): {num_node}")
    if max_test_id >= num_node:
        print("⚠️ Test set has item ID exceeding num_node — this will cause an indexing error!")


    # Temporary: explicitly without global graph:
    model = trans_to_cuda(CombineGraph(opt, num_node, edge_index=edge_index, features=x))


    print(opt)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
