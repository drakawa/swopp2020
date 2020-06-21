# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:00:05 2020

@author: ryuut
"""


import networkx as nx
import os.path
from sys import exit
from collections import defaultdict as dd
from collections import Counter
import random
import itertools as it
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim

import torch.autograd.profiler

from collections import Counter
import numpy as np

import torch
import torch.nn.functional as F

import sys

class RNN(nn.Module):
    def __init__(self, num_nodes, eps):
        super(RNN, self).__init__()

        self.num_nodes = num_nodes
        
        self.eps = eps

        self.a = Parameter(torch.randn(num_nodes, dtype=torch.double))
        self.beta = Parameter(torch.randn(num_nodes, dtype=torch.double))
        self.gamma = Parameter(torch.randn(num_nodes, dtype=torch.double))
        # self.a = Parameter(torch.randn(1, dtype=torch.double))
        # self.beta = Parameter(torch.randn(1, dtype=torch.double))
        # self.gamma = Parameter(torch.randn(1, dtype=torch.double))
        
        # print("self.a:", self.a)

        self.sigmoid = nn.Sigmoid()
        

    def sig_inv(self, i):
        return torch.clamp(torch.log(i/(1-i)), min=-sys.float_info.max, max=sys.float_info.max)

    def calc_Eit(self, Sit_prev, Eit_prev, Iit_prev, Rit_prev, a, beta, gamma, ts):
        return Eit_prev + (-a * ts)
    def calc_Iit(self, Sit_prev, Eit_prev, Iit_prev, Rit_prev, a, beta, gamma, ts):
        Eit_prev_weight = torch.log(self.my_clamp(a / (a - gamma) * (torch.exp(-gamma * ts) - torch.exp(-a * ts))))
        return torch.logsumexp(torch.stack((Eit_prev + Eit_prev_weight, Iit_prev + (-gamma * ts))), dim=0)
        # return Eit_prev * a / (a - gamma) * (torch.exp(-gamma * ts) - torch.exp(-a * ts)) +\
        #     Iit_prev * torch.exp(-gamma * ts)

    def calc_Rit(self, Sit_prev, Eit_prev, Iit_prev, Rit_prev, a, beta, gamma, ts):
        Eit_prev_weight = torch.log(self.my_clamp(1 + gamma / (a - gamma) * torch.exp(-a * ts) -\
                            a / (a - gamma) * torch.exp(-gamma * ts)))
        Iit_prev_weight = torch.log(self.my_clamp(1 - torch.exp(-gamma * ts)))
        return torch.logsumexp(torch.stack((Eit_prev + Eit_prev_weight, Iit_prev + Iit_prev_weight, Rit_prev)), dim=0)
        # return Eit_prev * (1 + gamma / (a - gamma) * torch.exp(-a * ts) -\
        #                     a / (a - gamma) * torch.exp(-gamma * ts)) +\
        #     Iit_prev * (1 - torch.exp(-gamma * ts)) + Rit_prev

    def calc_Sit(self, Sit_prev, Eit, Ijt, beta, Tijt):
        return Sit_prev + (-beta * torch.mv(Tijt, torch.exp(Ijt)))
        # return Sit_prev * torch.exp(-beta * torch.mv(Tijt, Ijt))   
    def calc_newEit(self, Sit_prev, Eit, Ijt, beta, Tijt):
        Sit_prev_weight = torch.log(self.my_clamp(1 - torch.exp(-beta * torch.mv(Tijt, torch.exp(Ijt)))))
        return torch.logsumexp(torch.stack((Sit_prev + Sit_prev_weight, Eit)), dim=0)
        # return Sit_prev * (1 - torch.exp(-beta * torch.mv(Tijt, Ijt))) + Eit
            
    def my_clamp(self, i):
        return torch.clamp(i, min=self.eps, max=1-self.eps)
    
    def forward(self, Ti, Cij, h_Sit0, h_Eit0, h_Iit0, h_Rit0, ASit, AEit, AIit, ARit):
        # print(Ti)
        a = self.sigmoid(self.a)
        beta = self.sigmoid(self.beta)
        gamma = self.sigmoid(self.gamma)
        # h_Sit = self.sigmoid(h_Sit0)
        # h_Eit = self.sigmoid(h_Eit0)
        # h_Iit = self.sigmoid(h_Iit0)
        # h_Rit = self.sigmoid(h_Rit0)
        # print(h_Sit0)
        # exit(1)
        
        h_Sit2 = h_Sit
        h_Eit2 = self.calc_Eit(h_Sit, h_Eit, h_Iit, h_Rit, a, beta, gamma, Ti)
        h_Iit2 = self.calc_Iit(h_Sit, h_Eit, h_Iit, h_Rit, a, beta, gamma, Ti)
        h_Rit2 = self.calc_Rit(h_Sit, h_Eit, h_Iit, h_Rit, a, beta, gamma, Ti)
        
        h_Sit3 = self.calc_Sit(h_Sit2, h_Eit2, h_Iit2, beta, Cij)
        h_Eit3 = self.calc_newEit(h_Sit2, h_Eit2, h_Iit2, beta, Cij)
        h_Iit3 = h_Iit2
        h_Rit3 = h_Rit2
        # print(h_Rit3)
        # exit(1)

        # return self.my_clamp(h_Sit3), self.my_clamp(h_Eit3), self.my_clamp(h_Iit3), self.my_clamp(h_Rit3)
        return h_Sit3, h_Eit3, h_Iit3, h_Rit3

class ReadEdgeT:
    def __init__(self, dirname, filename):
        self.dirname, self.filename = dirname, filename
    def read_edge_t(self):
        with open(os.path.join(self.dirname, self.filename), "r") as f:
            for l in f.readlines():
                t, s, d = list(map(int, l.rstrip().split()))
                yield s, d, t
                
class RunSim:
    def __init__(self, individuals, edge_ts, a, beta, gamma, i_init, seed=None):
        self.individuals = individuals
        self.edge_ts = edge_ts
        self.a = a
        self.beta = beta
        self.gamma = gamma
        self.i_init = i_init
        
        self.tmin = min(self.edge_ts)
        self.tmax = max(self.edge_ts)
        
        if seed != None:
            random.seed(seed)
            
        init_i_inds = random.sample(self.individuals, int(round(len(self.individuals) * self.i_init)))
        
        self.states = dict()
        init_state = dict()
        for i in self.individuals:
            if i in init_i_inds:
                init_state[i] = "I"
            else:
                init_state[i] = "S"
        self.states[self.tmin-1] = init_state
        print(Counter(self.states[self.tmin-1].values()))
    
    def runsim(self):
        for t in range(self.tmin, self.tmax + 1):
            tmp_state = self.states[t-1].copy()
            to_infect = list()
            for s, d in self.edge_ts[t]:
                if tmp_state[s] == "S" and tmp_state[d] == "I" and random.random() < self.beta:
                    to_infect.append(s)
                elif tmp_state[s] == "I" and tmp_state[d] == "S" and random.random() < self.beta:
                    to_infect.append(d)
            
            for i in list(tmp_state.keys()):
                if tmp_state[i] == "E" and random.random() < self.a:
                    tmp_state[i] = "I"
                elif tmp_state[i] == "I" and random.random() < self.gamma:
                    tmp_state[i] = "R"
            for i in to_infect:
                tmp_state[i] = "E"
            self.states[t] = tmp_state
            
            tmp_counter = Counter(self.states[t].values())
            if t % 100 == 0:
                print(t)
                print(tmp_counter)
            if "E" not in tmp_counter and "I" not in tmp_counter:
                break
                
        return self.states

class GenLayer:
    def __init__(self, states, rest_ets, individuals, tmin, tmax, stepsize):
        self.states = states
        self.rest_ets = rest_ets
        self.individuals = individuals
        self.stepsize = stepsize
        self.tmin = tmin
        self.tmax = tmax
        
        self.tmp_t = tmin
        self.tmp_G = None
        self.T_i = {i:0 for i in self.individuals}
        
    def _process_one_step(self):
        if self.tmp_t > self.tmax:
            return False
        
        G = nx.Graph()
        for t in range(self.tmp_t, self.tmp_t + self.stepsize):
            tmp_edges = self.rest_ets[t]
            for s, d in tmp_edges:
                if not G.has_edge(s, d):
                    G.add_edge(s, d, time=1)
                else:
                    G[s][d]["time"] += 1
            
        self.tmp_t += self.stepsize
        self.tmp_G = G
        
        return True
        
    def gen_layer(self):

        C_ij = {i:{j:0 for j in self.individuals} for i in self.individuals}
        nxt_T_i = {i:self.stepsize for i in self.individuals}
        A_i = {i:[0,0,0] for i in self.individuals}
        Ah_i = {i:[0,0,0] for i in self.individuals}
        
        processed = False
        if self.tmp_G == None:
            self._process_one_step()
        prev_G = self.tmp_G
        processed = self._process_one_step()
        
        if not processed:
            return -1, -1, -1, -1
        
        while processed and len(set(prev_G.nodes()) & set(self.tmp_G.nodes())) == 0:
            prev_G = nx.compose(prev_G, self.tmp_G)
            plus_nodes = set(self.individuals) - set(self.tmp_G.nodes())

            for pn in plus_nodes:
                nxt_T_i[pn] += self.stepsize
            for tn in self.tmp_G.nodes():
                self.T_i[tn] += nxt_T_i[pn]
                nxt_T_i[tn] = self.stepsize
            processed = self._process_one_step()
            
        for i in self.individuals:
            
            timestamp_i = self.tmp_t - nxt_T_i[i] - self.stepsize

            i_state = self.states[timestamp_i][i]

            if i_state == "S":
                A_i[i] = [1, 0, 0]
                Ah_i[i] = [1, 0, 0, 0]
            elif i_state == "E":
                A_i[i] = [1, 0, 0]
                Ah_i[i] = [0, 1, 0, 0]
            elif i_state == "I":
                A_i[i] = [0, 1, 0]
                Ah_i[i] = [0, 0, 1, 0]
            elif i_state == "R":
                A_i[i] = [0, 0, 1]
                Ah_i[i] = [0, 0, 0, 1]
            else:
                print("error:", i_state, A_i[i])
                exit(1)
            
            for j in self.individuals:
                if prev_G.has_edge(i, j):
                    C_ij[i][j] = prev_G[i][j]["time"]
                    
        tmp_T_i = self.T_i
        self.T_i = nxt_T_i
        
        return C_ij, tmp_T_i, A_i, Ah_i
            
if __name__ == "__main__":
    dirname = "contact"
    filename = "tij_InVS13.dat"
    
    
    print(os.path.join(dirname, filename))
 
    individuals = set()
    ret = ReadEdgeT(dirname, filename)
    rest_ets = dd(list)
    for s, d, t in ret.read_edge_t():
        rest_ets[t].append((s, d))
        individuals |= set([s, d])
        
    tmp_count = 0

    print(len(rest_ets.keys()))
    print(max(rest_ets), min(rest_ets))
    print(len(individuals))
    
    a, beta, gamma, i_init = 0.05, 0.15, 0.0005, 0.1
    seed = 2
    runsim = RunSim(individuals, rest_ets, a, beta, gamma, i_init, seed)
    result = runsim.runsim()
    
    stepsize = 10
    tmax, tmin = max(rest_ets), min(rest_ets)
    tmax = min([tmax, max(list(result.keys()))])
    
    gen_layer = GenLayer(result, rest_ets, individuals, tmin, tmax, stepsize)
    
    cta_count = 0
    node_idx = dict([(idx, ind) for idx, ind in enumerate(sorted(list(individuals)))])
    print(node_idx)
    # exit(1)
    Cijl, Til, Ail, Ahil = list(), list(), list(), list()
    while True:
        cta_count += 1
        C, T, A, Ah = gen_layer.gen_layer()
        if C != -1:
            pass
            # print(A)
        else:
            break
        # print(node_idx.keys())
        Cijl.append([[C[node_idx[i]][node_idx[j]] for j in node_idx.keys()] for i in node_idx.keys()])
        if T != None:
            Til.append([T[node_idx[i]]for i in node_idx.keys()])
        Ail.append([A[node_idx[i]] for i in node_idx.keys()])
        Ahil.append([Ah[node_idx[i]] for i in node_idx.keys()])

    print(len(Cijl), len(Til), len(Ail), len(Ahil))
    Cijl = np.array(Cijl)
    Til = np.array(Til)
    Ail = np.array(Ail)
    Ahil = np.array(Ahil)

    print(Cijl.shape, Til.shape, Ail.shape, Ahil.shape)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("device:", device)

    Cijl = torch.tensor(Cijl, device=device, dtype=torch.double)
    Til = torch.tensor(Til, device=device, dtype=torch.double)
    Ail = torch.tensor(Ail, device=device, dtype=torch.double)
    Ahil = torch.tensor(Ahil, device=device, dtype=torch.double)

    learning_rate = 0.05
    num_eps = sys.float_info.min
    net = RNN(len(individuals), num_eps)
    print(net)
    net.to(device)
    params = list(net.parameters())
    print(params)
    # exit(1)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    num_layers = len(Cijl)
    num_individuals = len(individuals)
    num_epochs = 200
    
    layer_train_mask = [True for _ in range(num_layers)]

    # ind_train_mask = [bool(Ail[0][n][0] == 1 and Ail[-1][n][0] == 0) for n in range(num_individuals)]
    ind_train_mask = [True for n in range(num_individuals)]
    
    layer_test_mask = [not l for l in layer_train_mask]
    ind_test_mask = [not l for l in ind_train_mask]
    
    print(ind_train_mask, sum(ind_train_mask))
    print(ind_test_mask)
    
    def sig_inv(i):
        return torch.clamp(torch.log(i/(1-i)), min=-sys.float_info.max, max=sys.float_info.max)

    print(num_individuals)
    print(num_layers)
    Ail_clamped = torch.clamp(Ail, min=num_eps, max=1-num_eps)

    for epoch_idx in range(num_epochs):
        num_losses = torch.tensor([0], device=device)

        sum_losses = 0.0

        h_Sit, h_Eit, h_Iit, h_Rit = torch.log(torch.clamp(Ail[0][:, 0], min=num_eps, max=1-num_eps)), torch.log(torch.clamp(torch.zeros(num_individuals, device=device, dtype=torch.double), min=num_eps, max=1-num_eps)),\
            torch.log(torch.clamp(Ail[0][:, 1], min=num_eps, max=1-num_eps)), torch.log(torch.clamp(Ail[0][:, 2], min=num_eps, max=1-num_eps))

        # print(h_Sit)
        # exit(1)
        optimizer.zero_grad()
        # print(net.a)
        # exit(1)
    
        # print(num_layers)
        # exit(1)
        for l in range(num_layers):

            h_Sit, h_Eit, h_Iit, h_Rit = net(Til[l], Cijl[l], h_Sit, h_Eit, h_Iit, h_Rit, Ail[l][:, 0], Ail[l][:, 0], Ail[l][:, 1], Ail[l][:, 2])

            h_Sit = h_Sit + torch.log(torch.clamp(Ail[l][:, 0], min=num_eps, max=1-num_eps))
            h_Eit = h_Eit + torch.log(torch.clamp(Ail[l][:, 0], min=num_eps, max=1-num_eps))
            h_Iit = h_Iit + torch.log(torch.clamp(Ail[l][:, 1], min=num_eps, max=1-num_eps))
            h_Rit = h_Rit + torch.log(torch.clamp(Ail[l][:, 2], min=num_eps, max=1-num_eps))

            # if l == 1:
            #     print(h_Sit)
            #     exit(1)
            # if layer_train_mask[l]:
            #     # h_SEit = torch.clamp(h_Sit + h_Eit, min=num_eps, max=1-num_eps)
            #     h_SEit = h_Sit + h_Eit

                # sum_losses -= torch.dot(h_SEit[ind_train_mask], Ail[l][:, 0][ind_train_mask])
                # sum_losses -= torch.dot(h_Iit[ind_train_mask], Ail[l][:, 1][ind_train_mask])
                # sum_losses -= torch.dot(h_Rit[ind_train_mask], Ail[l][:, 2][ind_train_mask])

        # print(h_Rit)
        sum_losses = -torch.mean(torch.logsumexp(torch.stack((h_Sit, h_Eit, h_Iit, h_Rit)), dim=0), dim=-1)
        print(sum_losses)
        # exit(1)
        # se_loss = torch.logsumexp(torch.stack((h_Sit, h_Eit)), dim=0)
        # i_loss = h_Iit[ind_train_mask]
        # r_loss = h_Rit[ind_train_mask]
        # sum_losses -= (se_loss + i_loss + r_loss)

        print("a:", net.a.data)
        # print(net.a.grad)
        # print(net.beta.grad)
        # print(net.gamma.grad)
        sum_losses.backward()

        # print(net.a.grad)
        # print(net.beta.grad)
        # print(net.gamma.grad)
        optimizer.step()
        # exit(1)

        # print("epoch_idx", epoch_idx, "sum_losses:", sum_losses, nn.Sigmoid()(net.a))
        print("epoch_idx", epoch_idx, "sum_losses:", float(sum_losses), nn.Sigmoid()(net.a))

    print(nn.Sigmoid()(net.a))
    print(nn.Sigmoid()(net.beta))
    print(nn.Sigmoid()(net.gamma))
    exit(1)
    # print(input_i, target_i)

    loss_SE2 = 0
    print("loss_SE2:", loss_SE2)
    # print("h_Sit:", h_Sit)
    loss_SE = criterion(h_Sit + h_Eit, Ail[0][:, 0])
    loss_SE2 += loss_SE * 2
    print("loss_SE:", loss_SE)
    print("loss_SE2:", loss_SE2)
    loss_SE2.backward()
    print(net.h_Sit.grad)
    optimizer.step()
    # print(torch.tensor([]))
    # print(h_Sit)
    exit(1)
        
    Gs = list()
    # result_offset = tmin - 1
    for tmp_t in range(tmin, tmax, stepsize):
        # if len(result) <= tmp_t - result_offset:
            # break
        # print(tmp_t)
        G = nx.Graph()
        # G.add_nodes_from(individuals)
        # print(result[tmp_t])
        # nx.set_node_attributes(G, result[tmp_t - result_offset], name="state")
        for t in range(tmp_t, tmp_t + stepsize):
            tmp_edges = rest_ets[t]
            for s, d in tmp_edges:
                if not G.has_edge(s, d):
                    G.add_edge(s, d, time=[t])
                else:
                    G[s][d]["time"].append(t)
        # print(tmp_t)
        # print(G.edges(data=True))
        # print(set(G.nodes()))
        Gs.append(G)
    print(len(Gs))
    # print(Gs[0].nodes(data=True))
    # exit(1)
    newGs = list()
    for G in Gs:
        if not nx.is_empty(G):
            newGs.append(G)
    print("remove_empty:", len(newGs))
    
    C_lij = list()
    A_li = list()

    C_lij.append(dd(lambda x: dd(int)))
    A_li.append(dd(list))    

    T_li = list()
    T_i = {i:stepsize for i in individuals}
    
    newGs2 = [Gs[0]]
    for G in Gs[1:]:
        tmp_G = newGs2[-1]
        if len(set(G.nodes()) & set(tmp_G.nodes())) == 0:
            newGs2[-1] = nx.compose(G, tmp_G)
            plus_nodes = set(individuals) - set(G.nodes())
            # print(plus_nodes)
            for pn in plus_nodes:
                T_i[pn] += stepsize
            for tn in G.nodes():
                T_i[tn] = stepsize
        else:
            newGs2.append(G)
            T_li.append(T_i)
            T_i = {i:stepsize for i in individuals}
    # for G in newGs2:
    #     pass
    #     print(sorted(list(G.nodes())))
        # print(G.edges(data=True))
    # for i in range(1000):
    #     # print(T_li[i])
    #     tmp_counter = Counter(T_li[i].values())
    #     if not (len(tmp_counter.keys()) == 1 and list(tmp_counter.keys())[0] == 1):
    #         print(tmp_counter)
    print(len(newGs2))
    print(max(rest_ets), min(rest_ets), (max(rest_ets) - min(rest_ets)) // stepsize)
    # print(sorted(individuals), len(sorted(individuals)))
    print(len(sorted(individuals)))
    print(len(T_li))
    print(T_li[0])
    exit(1)
    while rest_ets:
        tmp_count += 1
        rest_ets_nxt = dd(list)
        print("hoge")
        G = nx.Graph()
        H = nx.Graph()
        for t in rest_ets:
            # print(t)
            tmp_edges = rest_ets[t]
            # print(tmp_edges)
            for s, d in tmp_edges:
                if s not in H and d not in H:
                    G.add_edge(s, d, time=[t])
                elif H.has_edge(s, d) and len(H[s]) == 1 and len(H[d]) == 1:
                    # it means (s, d) is isolated
                    G[s][d]["time"].append(t)
                else:
                    rest_ets_nxt[t].append((s, d))
                    
                if not H.has_edge(s, d):
                    H.add_edge(s, d)
        print("baka")
        print(G.edges(data=True))
        # print(H.edges())
        # print(H.nodes())
        # print(rest_ets_nxt)
        rest_ets = rest_ets_nxt
        print(len(rest_ets))
        if tmp_count > 300:
            exit(1)
        
    Gs = list()
    while rest_ets:
        G = nx.Graph()
        H = nx.Graph()
        new_H = H.copy()
        tmp_t = 0
        for s, d, t in rest_ets:
            if t > tmp_t:
                pass
    