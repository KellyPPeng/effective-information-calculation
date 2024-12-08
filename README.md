# effective-information-calculation
## 给定粗粒化方案
### 基于状态空间计算有效信息：calc_tpm
给定微观的状态转移矩阵（micro_tpm），即每个状态下一时刻转向所有状态的概率。以4状态的微观系统为例:

    from ce import *
    micro_tpm = torch.tensor([[1/3, 1/3, 1/3, 0], [1/3, 1/3, 1/3, 0], [1/3, 1/3, 1/3, 0], [0, 0, 0, 1]])
    
定义微观状态到宏观状态的映射方案（group），例如前三个微观状态映射到同一个宏观态，最后一个微观状态单独映射到另一个宏观态：

    group = [(0, 1, 2), (3,)]
    
使用calc_tpm可以得到宏观的概率转移矩阵，以及宏观和微观的有效信息值，二者相减得到因果涌现值。

    micro_ei, macro_ei, macro_tpm = calc_tpm(micro_tpm, group)
    CE = macro_ei - micro_ei

    
### 基于变量空间计算有效信息：calc_bn_ei
给定微观布尔元素的个数，定义微观布尔机制（micro_mech）。例如，当元素接收的输入为00，01或10时，则下一时刻该元素有0.7的概率为0，0.3的概率为1；当接收的输入为11时，下一时刻该元素为1，定义如下：

    from ce import *
    micro = 4
    micro_mech = {'00': [0.7, 0.3], '01': [0.7, 0.3], '10': [0.7, 0.3], '11': [0, 1]}

定义每个元素接受来自哪两个元素输入。例如，第1，2个元素来自第3，4个元素输入，第3，4个元素来自第1，2个元素输入，定义方法如下：

    edges = [(2, 3), (2, 3), (0, 1), (0, 1)]

粗粒化映射分两个部分，首先是元素的映射（elem_group），定义方法同基于状态空间的group。其次是机制的映射（mech_group），定义元素映射方法之后，每个宏观元素包含若干个微观元素，需要定义状态集合的映射。例如，微观00，01和10都映射为宏观的0状态，微观的11映射为宏观1状态，定义如下：

    elem_group = [(0, 1), (2, 3)]
    mech_group = {'00': '0', '01': '0', '10': '0', '11': '1'}
    
最后，使用calc_bn_ei得到微观、宏观的TPM，以及各自的有效信息值：

    micro_ei, macro_ei, S_m, S_M = calc_bn_ei(micro_mech, micro, edges, elem_group, mech_group)
    CE = macro_ei - micro_ei


## 不给定粗粒化方案
### 贪婪算法
以上面基于变量空间的tpm为例，输入S_m。由于贪婪算法稳定性一般，易陷入局部最优解，所以采用多次学习的结果，取其中最大的归一化CE值对应的粗粒化方案。

    G = np.array(S_m)
    G = check_network(G)
    max_eff_gain = 0
    for i in range(100):
        CE = causal_emergence(G)
        Nmicro = CE['G_micro'].number_of_nodes()
        Nmacro = CE['G_macro'].number_of_nodes()
        eff_gain = CE['EI_macro']/np.log2(Nmacro) - CE['EI_micro']/np.log2(Nmicro)

        if eff_gain > max_eff_gain:
            max_eff_gain = eff_gain
            best_CE = CE
    G_macro = CE['G_macro']
    EI_macro = CE['EI_macro']
    mapping = CE['mapping']
            

### 谱分解算法
    G = np.array(S_m)
    G = check_network(G)
    CE = causal_emergence_spectral(G)
    
贪婪算法的具体实现路径可参考：https://wiki.swarma.org/index.php/%E5%A4%8D%E6%9D%82%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E5%9B%A0%E6%9E%9C%E6%B6%8C%E7%8E%B0

贪婪算法、谱分解代码（ce_net.py, ei_net.py）修改自：https://github.com/jkbren/einet

    
