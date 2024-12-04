import torch
import itertools

def Entropia(a):
    b = torch.nonzero(a > 0).squeeze(1)
    return -torch.sum(a[b]*torch.log2(a[b]))


def effective_information(G1):
    """
    Calculate EI value of transition probability matrix
    :param G1: TPM matrix
    :return: EI value
    """
    N = G1.shape[0]
    G = torch.zeros_like(G1)
    nozero_number = 0
    for i in range(N):
        if torch.sum(G1[i]) != 0:
            G[i] = G1[i] / torch.sum(G1[i])
            nozero_number += 1
    Wout = torch.zeros(N)
    Win = torch.zeros(N)

    for i in range(N):
        if torch.sum(G[i]) != 0:
            Wout[i] = torch.tensor(Entropia(G[i]))

    for i in range(N):
        for j in range(N):
            if G[i][j] != 0:
                Win[j] += G[i][j] / nozero_number
    Wout_average = torch.sum(Wout) / nozero_number
    Win_entropy = Entropia(Win)
    return Win_entropy - Wout_average


def calc_tpm(micro_tpm, group):
    """
    Calculate the effective information of TPM in micro and macro level, given the micro mechanism and
    coarse-graining methods.

    Parameters
    ----------
    micro_tpm: torch.tensor
    group: the number of micro elements

    Returns
    -------
    the EI value of micro and macro TPM.

    Example
    -------
    micro_tpm = torch.tensor([[1/3, 1/3, 1/3, 0], [1/3, 1/3, 1/3, 0], [1/3, 1/3, 1/3, 0], [0, 0, 0, 1]])
    group = [(0, 1, 2), (3,)]
    micro_ei, macro_ei = calc_tpm(micro_tpm, group)

    """
    Nmi = micro_tpm.shape[0]
    Nma = len(group)
    mapping = torch.zeros((Nmi, Nma), dtype=torch.float)

    for j in range(Nma):
        for i in group[j]:
            mapping[i][j] = 1
    macro_tpm = mapping.t() @ micro_tpm @ mapping

    micro_ei = effective_information(micro_tpm)
    macro_ei = effective_information(macro_tpm)

    return micro_ei, macro_ei



def calc_bn_ei(micro_mech, micro, edges, elem_group, mech_group):
    """
    Calculate the effective information of boolean network in micro and macro level, given the micro mechanism and
    coarse-graining methods.

    Parameters
    ----------
    micro_mech: boolean function, e.g., {'00': [0.7, 0.3], '01': [0.7, 0.3], '10': [0.7, 0.3], '11': [0, 1]} the
            value means the probability of transferring to '0' and '1', correspondingly.
    micro: the number of micro elements
    edges: a list of tuple. the list means the input micro elements. e.g., edges[1] = (2, 3) means
            the input of elements[1] is [2] and [3]
    elem_group: the coarse-graining methods of micro elements, e.g., [(0, 1), (2, 3)] means coarse-graining
            elements[0],[1] to be a macro element, and the [2] and [3] to be another.
    mech_group: the coarse-graining method of micro_mech, e.g.,  mech_group = {'00': '0', '01': '0', '10': '0',
             '11': '1'}

    Returns
    -------
    the EI value of micro and macro level.

    Example
    -------
    micro_mech = {'00': [0.7, 0.3], '01': [0.7, 0.3], '10': [0.7, 0.3], '11': [0, 1]}
    micro = 4
    edges = [(2, 3), (2, 3), (0, 1), (0, 1)]
    elem_group = [(0, 1), (2, 3)]
    mech_group = {'00': '0', '01': '0', '10': '0', '11': '1'}
    micro_ei, macro_ei = calc_bn_ei(micro_mech, micro, edges, elem_group, mech_group)

    """
    Nmi = 2 ** micro
    node_list = list(itertools.product(['0', '1'], repeat=micro))
    S_m = torch.zeros((Nmi, Nmi), dtype=torch.float)
    for i, pre in enumerate(node_list):
        for j, tmp in enumerate(node_list):
            tr_pr = 1
            for z in range(micro):
                srce1 = pre[edges[z][0]]
                srce2 = pre[edges[z][1]]
                tr_pr *= micro_mech[srce1 + srce2][int(tmp[z])]
            S_m[i][j] = tr_pr

    unique = set(mech_group.values())
    Nma = len(unique) ** len(elem_group)
    group = torch.zeros((Nmi, Nma), dtype=torch.float)
    for i, item in enumerate(node_list):
        Sma = ''
        for elem in elem_group:
            Smi = ''.join(item[idx] for idx in elem)
            Sma += mech_group[Smi]
        group[i, int(Sma, base=len(unique))] = 1

    S_M = group.t() @ S_m @ group
    S_M = S_M / torch.sum(S_M, axis=1).unsqueeze(1)

    micro_ei = effective_information(S_m)
    macro_ei = effective_information(S_M)

    return micro_ei, macro_ei