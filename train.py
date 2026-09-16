import gzip
import argparse
import time
import datetime
import os
import pickle
import operator
import numpy as np
import torch
import json

# --- UPDATED IMPORT ---
from model.mgu_gnn import Data, split_validation, SessionGraph
from build_global_graph import build_global_graph

USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('--dataset', default='amazon_beauty', help='dataset name')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='learning rate decay step')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='early stop patience')
parser.add_argument('--nonhybrid', action='store_true', help='only use global preference')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='validation split')
# --- NEW: global graph hyperparameters ---
parser.add_argument('--global_window', type=int, default=3, help='co-occurrence window size for the global graph')
parser.add_argument('--global_max_neighbors', type=int, default=12, help='neighbors kept per item in the global graph')
opt = parser.parse_args()

def read_jsonl(file_path):
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            while True:
                try:
                    line = f.readline()
                    if not line: 
                        break
                    yield line
                except EOFError:
                    print(f"\n[WARNING] Hit a broken line! {file_path} is an incomplete download.")
                    print("Salvaging the data we successfully read so far...")
                    break 
    except gzip.BadGzipFile:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line

reviews_file_path = 'Beauty_and_Personal_Care.jsonl' 

# ==========================================
# DATA PROCESSING
# ==========================================
if not os.path.exists('dataset/amazon_beauty/train.txt'):
    print("-- Loading local Amazon Reviews JSONL Dataset @ %ss" % datetime.datetime.now())
    
    if not os.path.exists(reviews_file_path):
        print(f"\n[ERROR] Cannot find {reviews_file_path}!")
        exit()

    sess_clicks = {}
    sess_date = {}

    for line in read_jsonl(reviews_file_path):
        try:
            row = json.loads(line)
            sessid = row.get('user_id')
            item = row.get('parent_asin')
            date = row.get('timestamp') 
            
            if not sessid or not item or not date:
                continue

            if sessid in sess_clicks:
                sess_clicks[sessid].append((item, date))
                sess_date[sessid] = max(sess_date[sessid], date)
            else:
                sess_clicks[sessid] = [(item, date)]
                sess_date[sessid] = date
        except json.JSONDecodeError:
            continue

    print("-- Processing interactions @ %ss" % datetime.datetime.now())
    for sessid in sess_clicks:
        sorted_clicks = sorted(sess_clicks[sessid], key=lambda x: x[1])
        sess_clicks[sessid] = [click[0] for click in sorted_clicks]

    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            del sess_clicks[s]
            del sess_date[s]

    iid_counts = {}
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1

    for s in list(sess_clicks):
        curseq = sess_clicks[s]
        filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
        if len(filseq) < 2:
            del sess_clicks[s]
            del sess_date[s]
        else:
            sess_clicks[s] = filseq

    dates = list(sess_date.items())
    maxdate = max(dates, key=lambda x: x[1])[1]
    time_offset = (86400 * 7 * 1000) if maxdate > 1e11 else (86400 * 7)
    splitdate = maxdate - time_offset

    print('Splitting date', splitdate)
    tra_sess = filter(lambda x: x[1] < splitdate, dates)
    tes_sess = filter(lambda x: x[1] > splitdate, dates)

    tra_sess = sorted(tra_sess, key=operator.itemgetter(1))
    tes_sess = sorted(tes_sess, key=operator.itemgetter(1))

    item_dict = {}
    item_ctr = 1

    def obtain_tra():
        global item_ctr
        train_ids, train_seqs, train_dates = [], [], []
        for s, date in tra_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
                else:
                    outseq += [item_ctr]
                    item_dict[i] = item_ctr
                    item_ctr += 1
            if len(outseq) < 2:
                continue
            train_ids += [s]
            train_dates += [date]
            train_seqs += [outseq]
        return train_ids, train_dates, train_seqs

    def obtain_tes():
        test_ids, test_seqs, test_dates = [], [], []
        for s, date in tes_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
            if len(outseq) < 2:
                continue
            test_ids += [s]
            test_dates += [date]
            test_seqs += [outseq]
        return test_ids, test_dates, test_seqs

    tra_ids, tra_dates, tra_seqs = obtain_tra()
    tes_ids, tes_dates, tes_seqs = obtain_tes()

    os.makedirs('dataset/amazon_beauty', exist_ok=True)
    id2asin = {v: k for k, v in item_dict.items()}
    with open('dataset/amazon_beauty/id2asin.json', 'w') as f:
        json.dump(id2asin, f)

    def process_seqs(iseqs, idates):
        out_seqs, out_dates, labs, ids = [], [], [], []
        for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
                out_dates += [date]
                ids += [id]
        return out_seqs, out_dates, labs, ids

    tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
    te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)

    tra = (tr_seqs, tr_labs)
    tes = (te_seqs, te_labs)

    pickle.dump(tra, open('dataset/amazon_beauty/train.txt', 'wb'))
    pickle.dump(tes, open('dataset/amazon_beauty/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('dataset/amazon_beauty/all_train_seq.txt', 'wb'))
    vocab_size = item_ctr
    pickle.dump(vocab_size, open('dataset/amazon_beauty/vocab_size.pkl', 'wb'))
    print(f'Done processing data. Total unique items mapped: {item_ctr - 1}')

# ==========================================
# MODEL EXECUTION
# ==========================================
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)

def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr

def main():
    train_data = pickle.load(open('dataset/amazon_beauty/train.txt', 'rb'))         
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('dataset/amazon_beauty/test.txt', 'rb'))
        
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    
    n_node = pickle.load(open('dataset/amazon_beauty/vocab_size.pkl', 'rb'))
    print(f"Loaded vocab size (n_node): {n_node}")

    feature_matrix_path = 'dataset/amazon_beauty/item_image_features.pt'
    if not os.path.exists(feature_matrix_path):
        print("\n[WARNING] Could not find extracted images. Please run model/vit_encoder.py first!")
        print("Exiting...")
        return
        
    print("Loading precomputed ViT features...")
    img_feature_matrix = torch.load(feature_matrix_path)
    img_feature_matrix = trans_to_cuda(img_feature_matrix)

    # ==========================================
    # GLOBAL SESSION GRAPH (cross-session graph) --- NEW ---
    # ==========================================
    global_idx_path = 'dataset/amazon_beauty/global_graph_idx.pt'
    global_weight_path = 'dataset/amazon_beauty/global_graph_weight.pt'

    if not os.path.exists(global_idx_path):
        print("Building global cross-session item graph (one-time)...")
        all_train_seq = pickle.load(open('dataset/amazon_beauty/all_train_seq.txt', 'rb'))
        neighbor_idx, neighbor_weight = build_global_graph(
            all_train_seq, n_node, opt.global_window, opt.global_max_neighbors
        )
        torch.save(neighbor_idx, global_idx_path)
        torch.save(neighbor_weight, global_weight_path)
        print(f"Saved global graph: {neighbor_idx.shape} neighbors/item")

    print("Loading global graph...")
    neighbor_idx = trans_to_cuda(torch.load(global_idx_path))
    neighbor_weight = trans_to_cuda(torch.load(global_weight_path))

    model = trans_to_cuda(SessionGraph(opt, n_node, img_feature_matrix, neighbor_idx, neighbor_weight))

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
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()