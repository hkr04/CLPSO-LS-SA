import sys
sys.path.insert(0, './')

# from CEC2013 import functions
from cec2013single.cec2013 import Benchmark
import random
import numpy as np
import matplotlib.pyplot as plt

benchmark = Benchmark()

class best_ind:
    # 仅记录位置和适应度函数
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], individual) or isinstance(args[0], best_ind):
                self.x = args[0].x[:]
                self.f = args[0].f
        elif len(args) == 2:
            if isinstance(args[0], individual) and isinstance(args[1], int):
                self.x = args[0].x[:]
                self.f = args[0].f
                self.id = args[1]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

class individual:
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], individual): # 复制
                self.x = args[0].x[:]
                self.v = args[0].v[:]
                self.f = args[0].f
                self.pbest = best_ind(self)
            elif isinstance(args[0], list) or isinstance(args[0], np.ndarray): # 初始化
                self.x = np.array(args[0][:], dtype=float)
                self.v = np.random.standard_normal(self.x.size)
                self.f = None
                self.pbest = None
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

class population(list):
    def __init__(self, *args):
        for arg in args[0]:
            self.append(individual(arg))

class problem:
    def __init__(self, pop_size, ind_size, x_min, x_max, fitness_function, update, n_gen, \
                 fitness_max=False, callback=None, label=None, Max_FES=None, **kwargs):
        self.pop_size = pop_size
        self.ind_size = ind_size
        self.x_min = x_min
        self.x_max = x_max
        if isinstance(fitness_function, int): # 标准库
            self.fitness_function, self.gt = get_std(ind_size, fitness_function)
        else: 
            self.fitness_function = fitness_function
            self.gt = None
        self.update = update
        self.n_gen = n_gen
        self.fitness_max = fitness_max
        self.callback = callback
        self.label = label
        self.Max_FES = 1e18 if Max_FES is None else Max_FES
        self.exp_cnt = 0
        for name, value in kwargs.items():
            setattr(self, name, value)
                
    def train(self, exp_cnt=1):
        self.val_seq = [] # 每一行存储一次实验的信息(适应度值），列为迭代次数
        self.err_seq = [] # 每一行存储一次实验的信息(绝对误差），列为迭代次数
        self.ind_seq = [] # 每一行存储一次实验的信息(最优个体），列为迭代次数
        self.exp_cnt = exp_cnt
        for _ in range(exp_cnt):
            # 初始化
            self.fitness_function.reset()
            self.gbest = None
            self.pop = gen_pop(pop_size=self.pop_size, n_dim=self.ind_size, x_min=self.x_min, x_max=self.x_max)
            for ind in self.pop:
                ind.f = self.fitness_function(ind.x)
                ind.pbest = best_ind(ind)
            for i, ind in enumerate(self.pop):
                if self.gbest is None or ((ind.f < self.gbest.f) ^ self.fitness_max):
                    self.gbest = best_ind(ind, i)
            self.gbest_seq = [self.gbest]
            # 训练
            for self.cnt in range(1, self.n_gen+1):
                self.update(self)
                if self.fitness_function.count >= self.Max_FES:
                    self.gbest_seq.append(best_ind(self.gbest))
                    break
                if self.callback is not None:
                    self.callback(self)
                # gbest 可能在 callback 之后被更新
                self.gbest_seq.append(best_ind(self.gbest))
                if self.fitness_function.count >= self.Max_FES:
                    break
                
            self.val_seq.append([ind.f for ind in self.gbest_seq] + [self.gbest_seq[-1].f] * (self.n_gen - self.cnt))
            self.err_seq.append([abs(ind.f-self.gt) for ind in self.gbest_seq] + [abs(self.gbest_seq[-1].f-self.gt)] * (self.n_gen - self.cnt))
            self.ind_seq.append(self.gbest_seq[:] + [self.gbest_seq[-1]] * (self.n_gen - self.cnt))
        self.val_seq = np.array(self.val_seq) 
        self.err_seq = np.array(self.err_seq)
        print('--------------------')
        print(f'best: {self.best():.3e} worst: {self.worst():.3e} median: {self.median():.3e}')
        print(f'mean: {self.mean():.3e} std: {self.std():.3e}')
        print(f'FES: {self.fitness_function.count}')
        print('--------------------')
    
    def show(self, show_type='avg', show_errorbar=True):
        if show_type == 'avg':
            if show_errorbar:
                plt.errorbar(range(0, self.n_gen+1), self.val_seq.mean(axis=0), label='$95\%$conf', \
                             errorevery=10, yerr=np.percentile(self.val_seq, [2.5, 7.5], axis=0), capsize=5)
            else:
                plt.plot(range(0, self.n_gen+1), self.val_seq.mean(axis=0), label='average result')
        elif show_type == 'each':
            for i, seq in enumerate(self.val_seq):
                plt.plot(range(0, self.n_gen+1), seq, label='result ' + str(i+1))
        plt.xlabel('generation')
        plt.legend()
        plt.show()

    def max(self):
        return self.val_seq[:, -1].max(axis=0)
    
    def min(self):
        return self.val_seq[:, -1].min(axis=0)
    
    def median(self):
        return np.median(self.err_seq[:, -1], axis=0)
    
    def best(self):
        return self.err_seq[:, -1].min(axis=0)

    def worst(self):
        return self.err_seq[:, -1].max(axis=0)
    
    def mean(self):
        return self.err_seq[:, -1].mean(axis=0)
    
    def std(self):
        return self.err_seq[:, -1].std(axis=0)
    
def counter_wrapper(func):
    count = 0

    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        # print("Function {} has been called {} times.".format(func.__name__, count))
        wrapper.count = count
        return func(*args, **kwargs)
    
    def reset():
        nonlocal count
        count = 0

    wrapper.reset = reset

    return wrapper

def get_std(n_dim, fun_num):
    return counter_wrapper(benchmark.get_function(fun_num)), benchmark.get_info(fun_num, n_dim)['best']
    
def gen_pop(pop_size, n_dim, x_min, x_max):
    pop = []
    for _ in range(pop_size):
        pop.append(np.random.uniform(x_min, x_max, n_dim))
    return population(pop)

def callback(F):
#     print(f"generation {F.cnt}, gbest: {F.gbest.x}, fitness: {F.gbest.f}")
    print(f"generation {F.cnt}, fitness: {F.gbest.f}")    

def update(F, w_ini=0.9, w_end=0.4, c1=2, c2=2): # standard
    w = w_ini-(w_ini-w_end)*(F.cnt-1)/F.n_gen # linear
    for ind in F.pop:
        ind.v = w*ind.v + c1*random.random()*(ind.pbest.x-ind.x) + c2*random.random()*(F.gbest.x-ind.x)
        ind.x = ind.v + ind.x

def no_inertia_weight_upd(F):
    c1, c2 = 2, 2
    for ind in F.pop:
        ind.v = ind.v + c1*random.random()*(ind.pbest.x-ind.x) + c2*random.random()*(F.gbest.x-ind.x)
        ind.x = ind.v + ind.x

def show(*args, **kwargs):
    n_gen = args[0].n_gen
    exp_cnt = args[0].exp_cnt
    pop_size = args[0].pop_size
    fitness_max = args[0].fitness_max
    assert all(arg.n_gen == n_gen for arg in args), "Trained different times!"
    assert all(arg.exp_cnt == exp_cnt for arg in args), "Experimented different times!"
    assert all(arg.pop_size == pop_size for arg in args), "Different population sizes!"
    assert all(arg.fitness_max == fitness_max for arg in args), "Different goals of optimization!"
    # 清除图窗的内容，并重置轴属性
    plt.gca().clear()
    plt.gca().set_title('')
    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')
    plt.xlabel('Number of Iterations')
    if 'type' in kwargs:
        if kwargs['type'] == 'avg':
            for i, arg in enumerate(args):
                plt.plot(range(0, n_gen+1), 
                         arg.err_seq.mean(axis=0), 
                         label=arg.label if arg.label is not None else str(i+1))
            plt.ylabel('Average Error')
        elif kwargs['type'] == 'best':
            for i, arg in enumerate(args):
                plt.plot(range(0, n_gen+1), 
                         arg.err_seq.min(axis=0), 
                         label=arg.label if arg.label is not None else str(i+1))
            plt.ylabel('Best Error')
        elif kwargs['type'] == 'median':
            for i, arg in enumerate(args):
                plt.plot(range(0, n_gen+1), 
                         np.median(arg.err_seq, axis=0), 
                         label=arg.label if arg.label is not None else str(i+1))
            plt.ylabel('Median Error')
        else:
            raise ValueError(f"Invalid type {kwargs['type']}")
    else:
        for i, arg in enumerate(args):
            plt.plot(range(0, n_gen+1), arg.err_seq.mean(axis=0), label=arg.label if arg.label is not None else str(i+1))
        # plt.ylabel('Function Value')
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    plt.legend()
    if 'save_path' in kwargs:
        plt.savefig(kwargs['save_path'])
    else:
        plt.show()
    

if __name__ == "__main__":
    # 在标准函数 F1(Sphere) 上的训练示例
    test = problem(pop_size=40, ind_size=10, x_min=-100, x_max=100, fitness_function=1, update=update,\
                   n_gen=200, fitness_max=False, callback=None, label='PSO', Max_FES=None)
    test.train(50) # 独立训练 50 次
    print("results:", test.val_seq[:, -1])
    print(test.fitness_function.count)
    show(test, title='F1 in 10-D problem')

    # # 比较两种 PSO 在 F2 上的表现
    # test1 = problem(pop_size=40, ind_size=10, x_min=-100, x_max=100, fitness_function=2, update=no_inertia_weight_upd,\
    #                 n_gen=200, fitness_max=False, callback=None, termination_cond=None, label='no inertia weight PSO')
    # test1.train(20)
    # test2 = problem(pop_size=40, ind_size=10, x_min=-100, x_max=100, fitness_function=2, update=update,\
    #                 n_gen=200, fitness_max=False, callback=None, termination_cond=None, label='std PSO')
    # test2.train(20)
    # show(test1, test2, type='median')


