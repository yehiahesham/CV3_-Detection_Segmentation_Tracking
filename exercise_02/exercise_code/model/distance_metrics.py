import torch
from torch.nn import functional as F


def euclidean_squared_distance(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix (m x feat).
        input2 (torch.Tensor): 2-D feature matrix (n x feat).
    Returns:
        torch.Tensor: distance matrix (m x n).
    """
    m, n = input1.size(0), input2.size(0)
    ########################################################################
    # TODO:                                                                #
    # Compute a m x n tensor that contains the euclidian distance between  #
    # all m elements to all n elements. Each element is a feat-D vector.   #
    # distmat = ...                                                        #
    ########################################################################


    # distmat = torch.cdist(input1, input2)
    # input1 = input1[:, None, :]
    # input2 = input2[None, :, :]
    # distmat = torch.sqrt(torch.sum((input1 - input2) ** 2, dim=2))
    
    # distmat=[]
    # for one in input1:
    #     distmat.append([torch.norm(one - other_tensor) for other_tensor in input2])
    # distmat=torch.tensor(distmat)
    
    D = input1.shape[1]
    # input1 = input1.unsqueeze(1).expand(m, n, D)
    # input2 = input2.unsqueeze(0).expand(m, n, D)
    # distmat = torch.sum((input1[i]  - input2[i]) ** 2, dim=2)  
    
    distmat = torch.zeros((m, n), dtype=input1.dtype, device=input1.device)
    chunk_size=10
    for i in range(0, m, chunk_size):
        input1_chunk = input1[i:i + chunk_size]
        
        for j in range(0, n, chunk_size):
            input2_chunk = input2[j:j + chunk_size]
            # distmat[i:i + chunk_size, j:j + chunk_size] = \
            #     torch.sum((input1_chunk.unsqueeze(1) - input2_chunk.unsqueeze(0)) ** 2,dim=2)
            
            distances = input1_chunk.unsqueeze(1) - input2_chunk.unsqueeze(0)
            # distances = distances.clamp(min=1e-12)
            distances = distances **2
            distmat[i:i + chunk_size, j:j + chunk_size] = torch.sum(distances,dim=2)
            
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return distmat


def cosine_distance(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix (m x feat).
        input2 (torch.Tensor): 2-D feature matrix (n x feat).
    Returns:
        torch.Tensor: distance matrix (m x n).
    """
    ########################################################################
    # TODO:                                                                #
    # Compute a m x n tensor that contains the cosine similarity between   #
    # all m elements to all n elements. Each element is a feat-D vector.   #
    # NOTE: The provided vectors are NOT normalized. For normalized        #
    # features, the dot-product is equal to the cosine similariy.          #
    # see e.g. https://en.wikipedia.org/wiki/Cosine_similarity#Properties  #
    # cosine_similarity = ...                                              #
    ########################################################################

    device=input1.device
    cosine_similarity = F.cosine_similarity(input1.unsqueeze(1), input2.unsqueeze(0), dim=2).to(device)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    distmat = 1 - cosine_similarity
    return distmat

def compute_distance_matrix(input1, input2, metric_fn):
    """A wrapper function for computing distance matrix.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric_fn (func): A function computing the pairwise distance 
            of input1 and input2.
    Returns:
        torch.Tensor: distance matrix.
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1), f'Input size 1 {input1.size(1)}; Input size 2 {input2.size(1)}'

    return metric_fn(input1, input2)