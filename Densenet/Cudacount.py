import torch


print('cuda index:', torch.cuda.current_device())

print('gpu 개수:', torch.cuda.device_count())

print('graphic name:', torch.cuda.get_device_name())

cuda = torch.device('cuda')

print(cuda)