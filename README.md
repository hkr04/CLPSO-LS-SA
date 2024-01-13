# 基于 CLPSO 的 LS 改进：CLPSO-LS-SA
## 依赖环境
- cec2013single
- numpy
- matplotlib
- scipy

其中，cec2013single 的资源来自于[https://github.com/dmolina/cec2013single](https://github.com/dmolina/cec2013single)，为 CEC2013 官方测评框架 C 语言实现版本的非官方 Python 封装版本。

此处使用的版本修改了该资源中的部分代码，以确保`benchmark.get_info(fun_num, n_dim)['best']`字段与 [Problem Definitions and Evaluation Criteria for the CEC 
2013 Special Session on Real-Parameter Optimization](https://www.al-roomi.org/multimedia/CEC_Database/CEC2013/RealParameterOptimization/CEC2013_RealParameterOptimization_TechnicalReport.pdf) 中给出的最优解相符合。

可直接进入`cec2013single-master`文件夹，执行：
```cmd
python setup.py install
```
完成配置。

## 实验方式
确保当前运行目录为`exp`的上级目录，打开`exp/test.py`。

若直接运行，测评的函数为所有 $28$ 个 CEC2013 标准函数，维度为 $10$，测评的 PSO 变体为 $\text{PSO}, \text{CLPSO}, \text{CLPSO-LS}_{\text{BFGS}}(\text{CL}_{\text{BFGS}}),\text{CLPSO-LS}_{\text{Nelder–Mead}}(\text{CL}_{\text{N-M}})$ 以及改进的 $\text{CLPSO-LS}_{\text{Simulated Annealing}}(\text{CL}_{\text{SA}})$。默认情况下，输出图片结果在文件夹 fig 中，数据在根目录下的 result.csv 中。

## 参考文献
- [Liang J J, Qin A K, Suganthan P N, et al. Comprehensive learning particle swarm optimizer for global optimization of multimodal functions[J]. IEEE transactions on evolutionary computation, 2006, 10(3): 281-295.](https://ieeexplore.ieee.org/abstract/document/1637688/)
- [Cao Y, Zhang H, Li W, et al. Comprehensive learning particle swarm optimization algorithm with local search for multimodal functions[J]. IEEE Transactions on Evolutionary Computation, 2018, 23(4): 718-731.](https://ieeexplore.ieee.org/abstract/document/8561256/)
- [Liang J J, Qu B Y, Suganthan P N, et al. Problem definitions and evaluation criteria for the CEC 2013 special session on real-parameter optimization[J]. Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 2013, 201212(34): 281-295.](https://www.al-roomi.org/multimedia/CEC_Database/CEC2013/RealParameterOptimization/CEC2013_RealParameterOptimization_TechnicalReport.pdf)

