import itertools
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.autograd import grad
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from gridsample_grad2.cuda_gridsample import grid_sample_2d, grid_sample_3d
# F.grid_sample does not support gradient calcuation
import laplacian_py.laplacian_solver_py as laplacian_solver
import tinycudann as tcnn
import numpy as np
import rff


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
        self.V = max_val * torch.ones(self.res, dtype=torch.float32, device='cuda')
        self.obj_inds = None
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

    def reset(self):
        self.V = self.max_val * torch.ones(self.res, dtype=torch.float32, device='cuda')
        self.obj_inds = None

    def forward(self, x, boundary_types, boundary_conditions, calc_gradient=False):
        if self.obj_inds is None:
            self.obj_inds = boundary_types > 0
            self.V[self.obj_inds] = boundary_conditions[self.obj_inds]
        C_pos = F.relu(self.C) + self.cost_scale
        out = self.solve(self.V, C_pos, boundary_types, boundary_conditions, self.indexes, self.width)
        # out[out >= self.max_val] = .95 * self.max_val
        # out[obj_inds] = boundary_conditions[obj_inds]
        self.V = out.detach()
        J = None
        if calc_gradient:
            Jxyz = [torch.diff(out, dim=2, prepend=out[:, :, 0].unsqueeze(axis=2)),
                    torch.diff(out, dim=1, prepend=out[:, 0, :].unsqueeze(axis=1)),
                    torch.diff(out, dim=0, prepend=out[0, :, :].unsqueeze(axis=0))]

        out = interpolate_prediction(out.unsqueeze(axis=0).unsqueeze(axis=0), x)
        out = out.squeeze().squeeze()

        if calc_gradient:
            tmp = []
            for val in Jxyz:
                val_tmp = interpolate_prediction(val.unsqueeze(axis=0).unsqueeze(axis=0), x)
                while len(val_tmp.shape) > 3:
                    val_tmp = val_tmp.squeeze(axis=0)
                tmp.append(val_tmp)
            J = torch.stack(tmp, dim=3)

        return out, self.C, J


class LaplaceCostNet(nn.Module):
    def __init__(self, res, max_val, cost_scale, obj_width):
        super(LaplaceCostNet, self).__init__()
        self.max_val = max_val
        self.cost_scale = cost_scale
        self.res = res
        self.width = obj_width
        self.num_dims = len(self.res)
        self.V = max_val * torch.ones(self.res, dtype=torch.float32, device='cuda')
        self.obj_inds = None
        dims_linear = torch.zeros(self.num_dims, dtype=torch.int32)
        base = 1
        for ind, d in enumerate(res):
            dims_linear[ind] = base
            base = base * d
        self.indexes = torch.tensor([torch.sum(torch.tensor(v, dtype=torch.int32) * dims_linear) for v in
                                     list(itertools.product([-1, 0, 1], repeat=self.num_dims)) if
                                     not all(vi == 0 for vi in v)], dtype=torch.int32, device='cuda')

        self.C = nn.Parameter(self.cost_scale * torch.ones(self.res, dtype=torch.float32, device='cuda'))
        # grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, steps=100), torch.linspace(-1, 1, steps=100), indexing='ij')
        self.cost_net = tcnn.Network(
            n_input_dims=128,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 4,
            },
            # encoding_config={"otype": "Frequency",  # Component type.
            #                  "n_frequencies": 4  # Number of frequencies (sin & cos)
            #                  },
            # encoding_config={
            #     "otype": "Grid",  # Component type.
            #     "type": "Hash",  # Type of backing storage of the  grids. Can be "Hash", "Tiled" or "Dense".
            #     "n_levels": 16,  # Number of levels (resolutions)
            #     "n_features_per_level": 2,  # Dimensionality of feature vector  stored in each level's entries.
            #     "log2_hashmap_size": 19,
            #     # If type is "Hash", is the base-2 logarithm of the number of elements in each backing hash table.
            #     "base_resolution": 2,  # The resolution of the coarsest level is base_resolution^input_dims.
            #     "per_level_scale": 1.1,  # The geometric growth factor, i.e.
            #     # the factor by which the resolution of each grid is larger (per axis) than that of the preceding level.
            #     "interpolation": "Linear",  # How to interpolate nearby grid
            #     # lookups. Can be "Nearest", "Linear",
            #     # or "Smoothstep" (for smooth deri-
            #     # vatives).
            # },
        ).cuda()

        # self.cost_net = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     # nn.Linear(128, 128),
        #     # nn.ReLU(),
        #     nn.Linear(128, 1),
        # ).cuda()

        self.encoding = rff.layers.GaussianEncoding(.2, self.num_dims, 128//2).cuda()

        layer = LaplacianSolver()
        self.solve = layer.apply

    def reset(self):
        self.V = self.max_val * torch.ones(self.res, dtype=torch.float32, device='cuda')
        self.obj_inds = None

    def forward(self, x, boundary_types, boundary_conditions, calc_gradient=False):
        if self.obj_inds is None:
            self.obj_inds = boundary_types > 0
            self.V[self.obj_inds] = boundary_conditions[self.obj_inds]

        assert np.prod(self.res) == x.numel() // 2
        x2 = self.encoding((.5 + .5 * torch.reshape(x, [-1, 2])))
        # C = self.C
        C = self.cost_net(x2)
        C = torch.reshape(C, self.res)
        C.retain_grad()
        C_pos = F.relu(C)  # + self.cost_scale
        out = self.solve(self.V, C_pos, boundary_types, boundary_conditions, self.indexes, self.width)
        # out[out >= self.max_val] = .95 * self.max_val
        # out[obj_inds] = boundary_conditions[obj_inds]
        self.V = out.detach()
        J = None
        if calc_gradient:
            Jxyz = [torch.diff(out, dim=2, prepend=out[:, :, 0].unsqueeze(axis=2)),
                    torch.diff(out, dim=1, prepend=out[:, 0, :].unsqueeze(axis=1)),
                    torch.diff(out, dim=0, prepend=out[0, :, :].unsqueeze(axis=0))]

        out = interpolate_prediction(out.unsqueeze(axis=0).unsqueeze(axis=0), x)
        out = out.squeeze().squeeze()

        if calc_gradient:
            tmp = []
            for val in Jxyz:
                val_tmp = interpolate_prediction(val.unsqueeze(axis=0).unsqueeze(axis=0), x)
                while len(val_tmp.shape) > 3:
                    val_tmp = val_tmp.squeeze(axis=0)
                tmp.append(val_tmp)
            J = torch.stack(tmp, dim=3)

        return out, C, J


def interpolate_prediction(grid_pred, query):
    if len(grid_pred.shape) == 5:
        return grid_sample_3d(grid_pred, query, padding_mode='border', align_corners=True)
    else:
        return grid_sample_2d(grid_pred, query, padding_mode='border', align_corners=True)


def compute_loss(pred, target, C, cost_scale):
    loss = torch.sum((pred - target) ** 2) / pred.numel()
    neg_inds = C < 0
    if torch.any(neg_inds):
        # loss = loss + .00000001 * torch.sum(abs(C[neg_inds]))  # / torch.sum(neg_inds)
        loss = loss + .01 * torch.sum(abs(C[neg_inds]))  # / torch.sum(neg_inds)

    return loss


def calculate_gradient(grid_pred, query):
    """ calculate_gradient: calculate gradient of field at query point
    grid_pred : grid of predictions
    query: point to interpolate
    assumes axis_values are in [-1 1]
    """
    # mode='bilinear is required
    # out = F.grid_sample(grid_pred, query, mode='bilinear', padding_mode='zeros')
    if len(grid_pred.shape) == 5:
        grid_pred = grid_pred.permute([0, 1, 4, 3, 2])
    else:
        grid_pred = grid_pred.permute([0, 3, 2, 1])

    out = interpolate_prediction(grid_pred, query)
    dL_dout = torch.ones(out.shape, dtype=torch.float32, device='cuda')
    J1, J2 = grad([out], [grid_pred, query], grad_outputs=dL_dout, create_graph=True, retain_graph=True)

    return J1, J2
