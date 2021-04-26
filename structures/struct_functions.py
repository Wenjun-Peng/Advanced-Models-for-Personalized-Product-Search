import torch

def sequence_mask(add_lengths, max_len=None):
    lengths = add_lengths - 1
    batch_size=lengths.numel()
    max_len=max_len or lengths.max()
    return (torch.arange(0,max_len,device=lengths.device).type_as(lengths).unsqueeze(0).expand(batch_size,max_len).gt(lengths.unsqueeze(1)))

def sequence_mask_lt(lengths, max_len=None):
    batch_size=lengths.numel()
    max_len=max_len or lengths.max()
    return (torch.arange(0,max_len,device=lengths.device).type_as(lengths).unsqueeze(0).expand(batch_size,max_len).lt(lengths.unsqueeze(1)))

def sequence_mask_eq(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(batch_size,max_len).eq(lengths.unsqueeze(1)))

def L2Regularization(L2Weight, elements):
    loss = 0
    for e in elements:
        ve = e.view(-1)
        loss += torch.sum(torch.dot(ve,ve))
    return L2Weight * loss
