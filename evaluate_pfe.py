#!/usr/bin/python
# -*- encoding: utf-8 -*-


import sys
import os
import os.path as osp
import logging
import pickle
from tqdm import tqdm
import numpy as np
import torch
# from backbone import Network_D
from backbone_posterior import Network_D, Sigmanet
from torch.utils.data import DataLoader
from market1501 import Market1501



FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

PATH = './res/spherereid_pfe_with_sigmanet'
#'./res/spherereid/'

def negative_MLS(X_, Y, sigma_sq_X_, sigma_sq_Y, mean=False):
    # D = X.shape[1]
    # X = X.reshape(-1, 1, D)
    # Y = Y.reshape(1, -1, D)
    # sigma_sq_X = sigma_sq_X.reshape(-1, 1, D)
    # sigma_sq_Y = sigma_sq_Y.reshape(1, -1, D)
    # sigma_sq_fuse = sigma_sq_X + sigma_sq_Y
    # diffs = (X - Y) ** 2 / (1e-10 + sigma_sq_fuse) + np.log(sigma_sq_fuse+1e-6)
    # return diffs.sum(-1)
    # with open(PATH + '/embds_dist_mat.pkl', 'wb') as fw:
    diffs = []
    D = X_.shape[1]
    for i in range(0, X_.shape[0], 10):#
        # diffs_sub = []
        # for j in range(Y.shape[0]):
        #     tmp_mu = (X[i, :]-Y[j, :])**2
        #     # tmp_mu.type()
        #     tmp_sigma = sigma_sq_X[i, :] + sigma_sq_Y[j, :]
        #     tmp = tmp_mu/(1e-6+tmp_sigma)+np.log(tmp_sigma+1e-6)#84.71/87.18
        #     diffs_sub.append(tmp.sum(-1))
        print(i)
        s, e = i, i+10 if i+10<=X_.shape[0] else X_.shape[0]
        X = X_[s:e, :]
        sigma_sq_X = sigma_sq_X_[s:e, :]
        print('here1')
        X = X.reshape(-1, 1, D)
        Y = Y.reshape(1, -1, D)
        sigma_sq_X = sigma_sq_X.reshape(-1, 1, D)
        sigma_sq_Y = sigma_sq_Y.reshape(1, -1, D)
        print('here2')
        sigma_sq_fuse = sigma_sq_X + sigma_sq_Y
        print('here3')
        diffs_sub = (X - Y) ** 2 / (1e-10 + sigma_sq_fuse) + np.log(sigma_sq_fuse+1e-6)
        print('here4')
        diffs.append(diffs_sub.sum(-1))
        print(i)
        # pickle.dump(embd_res, fw)
    diffs = np.vstack(diffs)

    return diffs
    # with open(PATH+'/embds_dist_mat.pkl', 'rb') as fr:
    #     embd_res = pickle.load(fr)
    # return embd_res
def embed():
    ## load checkpoint
    res_pth = PATH
    mod_pth = osp.join('./res/spherereid', 'model_final.pkl')
    net = Network_D()
    net.load_state_dict(torch.load(mod_pth))
    net.cuda()
    net.eval()
    mod_pth = osp.join('./res/spherereid_pfe_with_sigmanet', 'model_final.pkl')
    sigmanet = Sigmanet()
    sigmanet.load_state_dict(torch.load(mod_pth))
    sigmanet.cuda()
    sigmanet.eval()

    ## data loader
    query_set = Market1501('/media/ivy/research/datasets/market1501/query',
            is_train = False)
    gallery_set = Market1501('/media/ivy/research/datasets/market1501/bounding_box_test',
            is_train = False)
    query_loader = DataLoader(query_set,
                        batch_size = 32,
                        num_workers = 4,
                        drop_last = False)
    gallery_loader = DataLoader(gallery_set,
                        batch_size = 32,
                        num_workers = 4,
                        drop_last = False)

    ## embed
    logger.info('embedding query set ...')
    query_pids = []
    query_camids = []
    query_embds = []
    for i, (im, _, ids) in enumerate(tqdm(query_loader)):
        embds = []
        for crop in im:
            crop = crop.cuda()
            mu, x_ = net(crop)
            sigma = sigmanet(x_)
            embd = torch.cat([mu, sigma], -1)
            embd = embd.detach().cpu().numpy()
            embds.append(embd)
        embed = sum(embds) / len(embds)
        pid = ids[0].numpy()
        camid = ids[1].numpy()
        query_embds.append(embed)
        query_pids.extend(pid)
        query_camids.extend(camid)
    query_embds = np.vstack(query_embds)#.astype(np.float16)
    query_pids = np.array(query_pids)
    query_camids = np.array(query_camids)

    # tmp = query_embds[:, :1024]
    # D = tmp.shape[1]

    logger.info('embedding gallery set ...')
    gallery_pids = []
    gallery_camids = []
    gallery_embds = []
    for i, (im, _, ids) in enumerate(tqdm(gallery_loader)):
        embds = []
        for crop in im:
            crop = crop.cuda()
            mu, x_ = net(crop)
            sigma = sigmanet(x_)
            embd = torch.cat([mu, sigma], -1)
            embd = embd.detach().cpu().numpy()
            embds.append(embd)
        embed = sum(embds) / len(embds)
        pid = ids[0].numpy()
        camid = ids[1].numpy()
        gallery_embds.append(embed)
        gallery_pids.extend(pid)
        gallery_camids.extend(camid)
    gallery_embds = np.vstack(gallery_embds)#.astype(np.float16)
    gallery_pids = np.array(gallery_pids)
    gallery_camids = np.array(gallery_camids)

    ## dump embeds results
    embd_res = (query_embds, query_pids, query_camids, gallery_embds, gallery_pids, gallery_camids)
    with open(PATH+'/embds.pkl', 'wb') as fw:
        pickle.dump(embd_res, fw)
    logger.info('embedding done, dump to: '+PATH+'/embds.pkl')

    return embd_res


def evaluate(embd_res, cmc_max_rank = 1):
    query_embds, query_pids, query_camids, gallery_embds, gallery_pids, gallery_camids = embd_res

    ## compute distance matrix
    logger.info('compute distance matrix')
    dist_mtx = np.matmul(query_embds, gallery_embds.T)
    dist_mtx = 1.0 / (dist_mtx + 1)
    n_q, n_g = dist_mtx.shape

    logger.info('start evaluating ...')
    indices = np.argsort(dist_mtx, axis = 1)
    matches = gallery_pids[indices] == query_pids[:, np.newaxis]
    matches = matches.astype(np.int32)
    all_aps = []
    all_cmcs = []
    for query_idx in tqdm(range(n_q)):
        query_pid = query_pids[query_idx]
        query_camid = query_camids[query_idx]

        ## exclude duplicated gallery pictures
        order = indices[query_idx]
        pid_diff = gallery_pids[order] != query_pid
        camid_diff = gallery_camids[order] != query_camid
        useful = gallery_pids[order] != -1
        keep = np.logical_or(pid_diff, camid_diff)
        keep = np.logical_and(keep, useful)
        match = matches[query_idx][keep]

        if not np.any(match): continue

        ## compute cmc
        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        all_cmcs.append(cmc[:cmc_max_rank])

        ## compute map
        num_real = match.sum()
        match_cum = match.cumsum()
        match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
        match_cum = np.array(match_cum) * match
        ap = match_cum.sum() / num_real
        all_aps.append(ap)

    assert len(all_aps) > 0, "NO QUERRY APPEARS IN THE GALLERY"
    mAP = sum(all_aps) / len(all_aps)
    all_cmcs = np.array(all_cmcs, dtype = np.float32)
    cmc = np.mean(all_cmcs, axis = 0)

    return cmc, mAP

def evaluate_pfe(embd_res, cmc_max_rank = 1):
    query_embds, query_pids, query_camids, gallery_embds, gallery_pids, gallery_camids = embd_res

    ## compute distance matrix
    logger.info('compute distance matrix')

    dist_mtx = negative_MLS(query_embds[:, :1024], gallery_embds[:, :1024], query_embds[:, 1024:], gallery_embds[:, 1024:])
    # dist_mtx = np.matmul(query_embds, gallery_embds.T)
    # dist_mtx = 1.0 / (dist_mtx + 1)
    n_q, n_g = dist_mtx.shape

    logger.info('start evaluating ...')
    indices = np.argsort(dist_mtx, axis = 1)
    matches = gallery_pids[indices] == query_pids[:, np.newaxis]
    matches = matches.astype(np.int32)
    all_aps = []
    all_cmcs = []
    for query_idx in tqdm(range(n_q)):
        query_pid = query_pids[query_idx]
        query_camid = query_camids[query_idx]

        ## exclude duplicated gallery pictures
        order = indices[query_idx]
        pid_diff = gallery_pids[order] != query_pid
        camid_diff = gallery_camids[order] != query_camid
        useful = gallery_pids[order] != -1
        keep = np.logical_or(pid_diff, camid_diff)
        keep = np.logical_and(keep, useful)
        match = matches[query_idx][keep]

        if not np.any(match): continue

        ## compute cmc
        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        all_cmcs.append(cmc[:cmc_max_rank])

        ## compute map
        num_real = match.sum()
        match_cum = match.cumsum()
        match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
        match_cum = np.array(match_cum) * match
        ap = match_cum.sum() / num_real
        all_aps.append(ap)

    assert len(all_aps) > 0, "NO QUERRY APPEARS IN THE GALLERY"
    mAP = sum(all_aps) / len(all_aps)
    all_cmcs = np.array(all_cmcs, dtype = np.float32)
    cmc = np.mean(all_cmcs, axis = 0)

    return cmc, mAP


if __name__ == '__main__':
    # embd_res = embed()
    with open(PATH+'/embds.pkl', 'rb') as fr:
        embd_res = pickle.load(fr)

    # cmc, mAP = evaluate(embd_res)#0.03444/0.05562
    cmc, mAP = evaluate_pfe(embd_res)
    print('cmc is: {}, map is: {}'.format(cmc, mAP))
