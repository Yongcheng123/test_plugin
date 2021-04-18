# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..losses import LossFunc
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import quadprog

L_KERNEL_REG = 0.00001


def conv_block_3d_down(in_dim, mid_dim, out_dim, activation):
    
    conv_down_1 = nn.Conv3d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv_down_1.weight, std = 0.1)
    conv_down_2 = nn.Conv3d(mid_dim, out_dim, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv_down_2.weight, std = 0.1)
    return nn.Sequential(
        conv_down_1, 
        nn.BatchNorm3d(mid_dim),
        activation,
        conv_down_2,
        nn.BatchNorm3d(out_dim),
        activation,                
        )


def conv_trans_block_3d(in_dim):
    return nn.ConvTranspose3d(in_dim, in_dim, kernel_size=2, stride=2, padding=0, output_padding=0)


def max_pooling_3d():
    return nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

    
def conv_block_3d_up(in_dim, out_dim, activation):
    conv_up_1 = nn.Conv3d(in_dim+out_dim, out_dim, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv_up_1.weight, std = 0.1)
    conv_up_2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv_up_2.weight, std = 0.1)
    return nn.Sequential(
        conv_up_1,
        nn.BatchNorm3d(out_dim),
        activation,
        conv_up_2,
        nn.BatchNorm3d(out_dim),
        activation,
        nn.Dropout(p=0.5, inplace=True))

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        activation = nn.ReLU()
        # Down sampling
        self.down_1 = conv_block_3d_down(1, 32, 64, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_3d_down(64, 64, 128, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_3d_down(128, 128, 256, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(256)
        self.up_1 = conv_block_3d_up(256, 128, activation)
        self.trans_2 = conv_trans_block_3d(128)
        self.up_2 = conv_block_3d_up(128, 64, activation)
#        
        # Output
        self.out = nn.Conv3d(64, 3, kernel_size=1, stride=1, padding=0)
        torch.nn.init.normal_(self.out.weight, std = 0.1)
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) 
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1) 
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2) 
        
        
        # Up sampling
        trans_1 = self.trans_1(down_3) 
        concat_1 = torch.cat([trans_1, down_2], dim=1)
        up_1 = self.up_1(concat_1) 
        trans_2 = self.trans_2(up_1) 
        concat_2 = torch.cat([trans_2, down_1], dim=1) 
        
        up_2 = self.up_2(concat_2) 
        # Output
        out = self.out(up_2) 
        return out
    
def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient (grad of current task), p-vector; 
        input:  memories (grad of previous tasks), (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    

def check_mem(_func):
    print('%s \t-- alloc: %dM, max alloc: %dM, cache: %dM, max cache: %dM' % (
	_func,
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
    ))

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


        
class Gem_UNet(nn.Module):
    def __init__(self, args):
        super(Gem_UNet, self).__init__()
        self.unet = UNet()

        self.opt = optim.Adam(self.parameters(), 1e-3, weight_decay = 0.01)
        self.lr_scheduler = StepLR(self.opt, step_size=100, gamma=0.95)
        self.margin = args.memory_strength

        self.n_memories = args.n_memories
        self.gpu = args.cuda


        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            #print("param:",param.data.numel())  # number of parameters per param
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), args.n_tasks) # store all grads
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
    def forward(self, x):
        #check_mem('before forward')
        output = self.unet(x)
        #check_mem('after forward')
        return output
    def observe(self, x, y, t, weight, past_data):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t


        #check_mem('before compute gradient on previous tasks')
        if len(self.observed_tasks) > 1:            
            for tt in range(len(self.observed_tasks) - 1):
                check_mem('before calculate past gradients')
                self.zero_grad()  
                past_task = self.observed_tasks[tt]
#

                ptloss = 0
                past_data_length = len(past_data[past_task])
                for past in past_data[past_task]:
                    if torch.cuda.memory_cached() / 1024 / 1024 > 14000:
                        torch.cuda.empty_cache()
                        continue
                    if torch.cuda.memory_allocated() / 1024 / 1024 > 9500:
                        past_data_length -= 1
                        continue
                    chains, batch, orig, coordinate_info = past
                    x_image, labels, pos_weight, class_dist = batch
                    #mrc, protein, chain = chains
                    if self.gpu:
                        x_image = x_image.cuda()
                        labels = labels.cuda()
                    #check_mem('before calculate '+chains[0]+' '+chains[1]+' '+chains[2]+' loss')
                    loss = LossFunc(pos_weight)
                    ptloss += loss(self.forward(x_image), labels)
                    check_mem('after calculate '+chains[0]+' '+chains[1]+' '+chains[2]+' loss')
                ptloss = ptloss / len(past_data[past_task])   
                
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)
#
        # now compute the grad on the current minibatch
        #check_mem('after compute gradient on previous tasks')
        
        #check_mem('before forward')
        x = self.forward(x)
        #check_mem('after forward')
        self.zero_grad()
        #check_mem('after zero')
        loss = LossFunc(weight)
        
        total_loss = loss(x, y)
        #check_mem('after loss')
        total_loss.backward()

        #check_mem('after loss backward')
        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])  # indx, is index of all previous tasks
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin) # self.margin= 0.5
                          
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)
        #check_mem('after calculate new graident')
        self.opt.step()

        self.lr_scheduler.step()
        self.lr_scheduler.get_lr()
        #check_mem('after update weight')

        return total_loss

