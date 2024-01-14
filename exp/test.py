import sys
sys.path.insert(0, './')

from MyToolbox.PSO import *
from functools import partial
from scipy.optimize import minimize
import random
import numpy as np
import csv
import os
import time

def update(F, w_ini=0.9, w_end=0.4, c1=2, c2=2): 
    w = w_ini-(w_ini-w_end)*(F.cnt-1)/F.n_gen # linear
    for ind in F.pop:
        for j in range(F.ind_size):
            ind.v[j] = w*ind.v[j] + c1*random.random()*(ind.pbest.x[j]-ind.x[j]) + c2*random.random()*(F.gbest.x[j]-ind.x[j])
            ind.x[j] = ind.v[j] + ind.x[j]
            ind.x[j] = max(F.x_min, min(F.x_max, ind.x[j]))

    for i, ind in enumerate(F.pop):
        if F.fitness_function.count >= F.Max_FES:
            break
        ind.f = F.fitness_function(ind.x)
        if (ind.f < ind.pbest.f) ^ F.fitness_max:
            ind.pbest = best_ind(ind)
        if (ind.f < F.gbest.f) ^ F.fitness_max:
            F.gbest = best_ind(ind, i)

def update_CLPSO(F, w_ini=0.9, w_end=0.4, c=1.49445): 
    if F.cnt == 1:
        F.flag = [F.gap] * F.pop_size
        F.Pc = [0.05 + 0.45 * (np.exp(10*i/(F.pop_size-1))-1) / (np.exp(10)-1) for i in range(F.pop_size)]
        F.f = [[0] * F.ind_size for _ in range(F.pop_size)]  

    for i, ind in enumerate(F.pop):
        # print(F.Pc[i], F.f[i])
        if F.flag[i] >= F.gap:
            F.flag[i] = 0
            Pc = F.Pc[i]
            for j in range(F.ind_size):
                if random.random() < Pc:
                    pos1, pos2 = random.sample(range(0, F.ind_size), 2)
                    F.f[i][j] = pos1 if F.pop[pos1].pbest.f < F.pop[pos2].pbest.f else pos2
                else:
                    F.f[i][j] = i
            no_other = True
            for j in range(F.ind_size):
                if F.f[i][j] != i:
                    no_other = False
                    break
            if no_other:
                pos = i
                while pos == i:
                    pos = random.randint(0, F.pop_size-1)
                d = random.randint(0, F.ind_size-1)
                F.f[i][d] = pos
                
    w = w_ini-(w_ini-w_end)*(F.cnt-1)/F.n_gen # linear
    for i, ind in enumerate(F.pop):
        for j in range(F.ind_size):
            ind.v[j] = w*ind.v[j] + c*random.random()*(F.pop[F.f[i][j]].pbest.x[j]-ind.x[j])
        ind.x = ind.v + ind.x

    for i, ind in enumerate(F.pop):
        if F.fitness_function.count >= F.Max_FES:
            break
        if min(ind.x) < F.x_min or max(ind.x) > F.x_max:
            continue
        ind.f = F.fitness_function(ind.x)
        if ind.f < ind.pbest.f:
            ind.pbest = best_ind(ind)
        else:
            F.flag[i] += 1
        if ind.f < F.gbest.f:
            F.gbest = best_ind(ind, i)

def update_CLPSO_LS(F, w_ini=0.9, w_end=0.4, c=1.49445): 
    if F.cnt == 1:
        F.qe0 = calc_qe(F)
        F.flag = [F.gap] * F.pop_size
        F.Pc = [0.05 + 0.45 * (np.exp(10*i/(F.pop_size-1))-1) / (np.exp(10)-1) for i in range(F.pop_size)]
        F.f = [[0] * F.ind_size for _ in range(F.pop_size)]  

    for i, ind in enumerate(F.pop):
        # print(F.Pc[i], F.f[i])
        if F.flag[i] >= F.gap:
            F.flag[i] = 0
            Pc = F.Pc[i]
            for j in range(F.ind_size):
                if random.random() < Pc:
                    pos1, pos2 = random.sample(range(0, F.ind_size), 2)
                    F.f[i][j] = pos1 if F.pop[pos1].pbest.f < F.pop[pos2].pbest.f else pos2
                else:
                    F.f[i][j] = i
            no_other = True
            for j in range(F.ind_size):
                if F.f[i][j] != i:
                    no_other = False
                    break
            if no_other:
                pos = i
                while pos == i:
                    pos = random.randint(0, F.pop_size-1)
                d = random.randint(0, F.ind_size-1)
                F.f[i][d] = pos
                
    w = w_ini-(w_ini-w_end)*(F.cnt-1)/F.n_gen # linear
    for i, ind in enumerate(F.pop):
        for j in range(F.ind_size):
            ind.v[j] = w*ind.v[j] + c*random.random()*(F.pop[F.f[i][j]].pbest.x[j]-ind.x[j])
        ind.x = ind.v + ind.x

    for i, ind in enumerate(F.pop):
        if F.fitness_function.count >= F.Max_FES:
            break
        if min(ind.x) < F.x_min or max(ind.x) > F.x_max:
            continue
        ind.f = F.fitness_function(ind.x)
        if ind.f < ind.pbest.f:
            ind.pbest = best_ind(ind)
        else:
            F.flag[i] += 1
        if ind.f < F.gbest.f:
            F.gbest = best_ind(ind, i)

def calc_qe(F, beta=1/3): # 计算 quasi-entropy 以判断是否进行 LS
    N = int(F.pop_size * beta)
    selected = np.random.choice(F.pop, size=N, replace=False)

    U_sum = 0
    for ind in selected:
        U = ind.pbest.f - F.gt
        U_sum += U
    
    qe = 0
    for ind in selected:
        U = ind.pbest.f - F.gt
        p = U / U_sum
        qe -= p * np.log(p) 
    return qe

def reach_GOB(F, theta=0.95, beta=1/3):
    F.qe = calc_qe(F, beta)
    return F.qe <= F.qe0 * theta

def LS(F, method):
    # print(f"generation {F.cnt}, fitness: {F.gbest.f}")   
    if reach_GOB(F) != True:
        return
    if F.fitness_function.count >= F.Max_FES:
        return
    result = minimize(fun=F.fitness_function, x0=F.gbest.x, method=method)
    ind = F.gbest.id
    if result.fun < F.gbest.f:
        F.pop[ind].x, F.pop[ind].f = result.x, result.fun
        F.pop[ind].pbest.x, F.pop[ind].pbest.f = result.x, result.fun
        F.gbest.x, F.gbest.f = result.x, result.fun
    F.terminate = True
        

def LS_SA(F, cooling_rate=0.95, Tf=0.001):
    if reach_GOB(F) != True:
        return
    if F.fitness_function.count >= F.Max_FES:
        return
    cur_x, cur_f = F.gbest.x, F.gbest.f
    T0 = F.qe0 - F.qe
    T = T0
    while T > Tf and F.fitness_function.count < F.Max_FES:
        scaling_factor = T / T0
        nxt_x = cur_x + np.random.uniform(low=scaling_factor*(F.x_min-cur_x), high=scaling_factor*(F.x_max-cur_x))
        nxt_f = F.fitness_function(nxt_x)
        if nxt_f < cur_f or random.random() < np.exp(-(nxt_f-cur_f)/T):
            cur_x, cur_f = nxt_x, nxt_f
        T *= cooling_rate
    ind, _ = max(enumerate(F.pop), key=lambda x: x[1].pbest.f)
    # print(ind)
    F.pop[ind].x, F.pop[ind].f = cur_x, cur_f
    F.pop[ind].pbest.x, F.pop[ind].pbest.f = cur_x, cur_f
    F.gbest.x, F.gbest.f = cur_x, cur_f

def test(std_func, problems):
    '''
    参数：
    - std_func(list): 测评函数在 CEC2013 中的编号，取值为 1~28
    - problems(list): 需要进行比较的 PSO 变种

    依照 std_func 中的顺序依次对 problems 中的变种进行评测
    默认将每个测评函数的结果图片存放于文件夹 fig 中，数据存放于根目录下的 result.csv 中（可自行调整路径）
    '''
    # 获取当前时间的字符串表示
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # 创建以当前时间命名的文件夹
    folder_name = 'res_' + current_time
    os.makedirs(folder_name)
    os.makedirs(folder_name + '/fig')
    labels = ['Func']
    for F in problems:
        label = F.label.translate(str.maketrans("", "", "${}"))
        labels += [label+'_best', label+'_median', label+'_mean', label+'_std']

    D = problems[0].ind_size
    with open(folder_name+'/result.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(labels)
        n_dim = problems[0].ind_size
        for fun_id in std_func:
            std = get_std(n_dim=n_dim, fun_num=fun_id)
            res = ['F'+str(fun_id)+'-10D']
            print(f'testing F{fun_id}')
            for F in problems:
                F.fitness_function, F.gt = std
                F.train(50)
                res += ['{:.3e}'.format(F.best()), '{:.3e}'.format(F.median()), '{:.3e}'.format(F.mean()), '{:.3e}'.format(F.std())]
            csv_writer.writerow(res)
            f.flush()
            show(*problems, \
                 title='F'+str(fun_id)+' in '+str(D)+'-D problem', \
                 save_path=folder_name+'/fig/avg' + str(fun_id).zfill(2) +".png", \
                 type='avg')
            show(*problems, \
                 title='F'+str(fun_id)+' in '+str(D)+'-D problem', \
                 save_path=folder_name+'/fig/best' + str(fun_id).zfill(2) +".png", \
                 type='best')
    f.close()

if __name__ == '__main__':
    random.seed()
    PSO = problem(pop_size=40, ind_size=10, x_min=-100, x_max=100, fitness_function=None, update=update,
                    n_gen=500, fitness_max=False, callback=None,
                    label='$PSO$')
    CLPSO = problem(pop_size=40, ind_size=10, x_min=-100, x_max=100, fitness_function=None, update=update_CLPSO,
                    n_gen=500, fitness_max=False, callback=None, Max_FES=40000,
                    label='$CLPSO$', gap=5)
    CLPSO_LS_BFGS = problem(pop_size=40, ind_size=10, x_min=-100, x_max=100, fitness_function=None, update=update_CLPSO_LS,
                    n_gen=500, fitness_max=False, callback=partial(LS, method='BFGS'), Max_FES=40000,
                    label='$CL_{BFGS}$', gap=5)
    CLPSO_LS_NM = problem(pop_size=40, ind_size=10, x_min=-100, x_max=100, fitness_function=None, update=update_CLPSO_LS,
                    n_gen=500, fitness_max=False, callback=partial(LS, method='Nelder-Mead'), Max_FES=40000,
                    label='$CL_{N-M}$', gap=5)
    CLPSO_LS_SA = problem(pop_size=40, ind_size=10, x_min=-100, x_max=100, fitness_function=None, update=update_CLPSO_LS,
                    n_gen=500, fitness_max=False, callback=LS_SA, Max_FES=40000,
                    label='$CL_{SA}$', gap=5)
    # test(range(1, 29), [PSO, CLPSO, CLPSO_LS_SA, CLPSO_LS_BFGS, CLPSO_LS_NM])
    test(list(range(21, 29))+list(range(1, 21)), [PSO, CLPSO, CLPSO_LS_SA, CLPSO_LS_BFGS, CLPSO_LS_NM])
    # test(range(1, 29), [PSO, CLPSO, CLPSO_LS_BFGS])
    # test(range(1, 29), [PSO, CLPSO, CLPSO_LS_SA])
    # test(range(1, 21), [PSO, CLPSO])