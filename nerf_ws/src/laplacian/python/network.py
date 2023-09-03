import itertools
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import laplacian_py.laplacian_solver_py as laplacian_solver


class LaplacianSolver(Function):
    @staticmethod
    def forward(ctx, V, C, boundary_types, boundary_conditions, indexes, width):
        num_dims = len(V.shape)
        objects_bounds = boundary_conditions * (boundary_types == 1) * (boundary_conditions > 0)
        if width > 0:
            if num_dims == 2:
                weight = torch.ones(1, 1, width, width, dtype=torch.float32, device='cuda')
                weight = weight / weight.numel()
                extra_cost = F.conv2d(objects_bounds.unsqueeze(dim=0), weight, bias=None, stride=1, padding='same',
                                      dilation=1, groups=1)

            else:
                weight = torch.ones(1, 1, width, width, width, dtype=torch.float32, device='cuda')
                weight = weight / weight.numel()
                extra_cost = F.conv3d(objects_bounds.unsqueeze(dim=0), weight, bias=None, stride=1, padding='same',
                                      dilation=1, groups=1)

            extra_cost = extra_cost * (boundary_types == 0)
            C = C + extra_cost
            C = torch.reshape(C, V.shape)

        X = torch.stack([V, boundary_types, boundary_conditions, C], axis=num_dims)
        intermediate = [C, boundary_types, boundary_conditions, indexes]

        def extract_value(T):
            if num_dims == 2:
                return T[:, :, 0].clone()
            else:
                return T[:, :, :, 0].clone()

        for i in range(5):
            intermediate.append(extract_value(X))
            laplacian_solver.forward(X, indexes)
            ctx.save_for_backward(*intermediate)

        out = extract_value(X)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dout):
        num_dims = len(dL_dout.shape)
        intermediate = ctx.saved_tensors
        grid_size = intermediate[0].shape
        dL_dC = torch.zeros(grid_size, dtype=torch.float32, device='cuda')

        C = intermediate[0]
        boundary_types = intermediate[1]
        boundary_conditions = intermediate[2]
        indexes = intermediate[3]
        dL_dC_i = torch.zeros(grid_size, dtype=torch.float32, device='cuda')
        dL_dV_i = torch.zeros(grid_size, dtype=torch.float32, device='cuda')
        dL_dout_i = dL_dout
        for V in intermediate[-1:3:-1]:
            X = torch.stack([V, boundary_types, boundary_conditions, C], axis=num_dims)
            laplacian_solver.backward(X, indexes, dL_dout_i, dL_dV_i, dL_dC_i)
            dL_dC += dL_dC_i
            dL_dout_i = dL_dV_i

        return dL_dV_i, dL_dC, None, None, None, None


class LaplaceNet(nn.Module):
    def __init__(self, res, max_val, cost_scale, obj_width):
        super(LaplaceNet, self).__init__()
        self.max_val = max_val
        self.cost_scale = cost_scale
        self.res = res
        self.width = obj_width
        self.num_dims = len(self.res)
        dims_linear = torch.zeros(self.num_dims, dtype=torch.int32)
        base = 1
        for ind, d in enumerate(res):
            dims_linear[ind] = base
            base = base * d
        self.indexes = torch.tensor([torch.sum(torch.tensor(v, dtype=torch.int32) * dims_linear) for v in
                                     list(itertools.product([-1, 0, 1], repeat=self.num_dims)) if
                                     not all(vi == 0 for vi in v)], dtype=torch.int32, device='cuda')

        self.C = nn.Parameter(self.cost_scale * torch.ones(self.res, dtype=torch.float32, device='cuda'))
        layer = LaplacianSolver()
        self.solve = layer.apply

    def forward(self, V, boundary_types, boundary_conditions):
        V = V.clone()
        obj_inds = boundary_types > 0
        V[obj_inds] = boundary_conditions[obj_inds]
        C_pos = F.relu(self.C)
        out = self.solve(V, C_pos, boundary_types, boundary_conditions, self.indexes, self.width)
        out[out >= self.max_val] = .95 * self.max_val
        out[obj_inds] = boundary_conditions[obj_inds]

        return out, C_pos


def compute_loss(pred, target, C_pos, cost_scale):
    loss = torch.sum((pred - target) ** 2) / pred.numel()
    min_val = cost_scale
    if torch.any(C_pos < min_val):
        loss = loss + 100 * torch.sum((C_pos[C_pos < min_val] - min_val) ** 2) / torch.sum(
            C_pos[C_pos < min_val])
    return loss
