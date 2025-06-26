import math

import numpy as np
import torch
import torch.nn as nn
from triton_utils import reduce, reshape_inputs

from mambular.ops.triton.kernel import mamba_tt  # type: ignore

try:
    import triton
    import triton.language as tl

except ImportError:
    triton = None
    tl = None
if triton is None or tl is None:
    raise ImportError("Triton is not installed. Please install Triton to use this module.")


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, c, delta, d=None):
        if x.dim() != 4:
            if x.dim() == 3 and a.dim() == 2:
                # reshape
                a, b, c, delta, x, _ = reshape_inputs(a, b, c, delta, x, None, rev=False)
            else:
                # throw error
                raise ValueError(f"Expected input with 3 or 4 dimensions, got {x.dim()}D tensor.")

        # Check shapes
        # assert b.shape == c.shape
        Ba, _, D, L = x.shape
        _, N, _, _ = b.shape

        BLOCKSIZE = 16

        BLOCKS = math.ceil(L / BLOCKSIZE)
        print("Sequence_length:", L)
        print("Number of Blocks:", BLOCKS)
        if BLOCKS % 2 != 0:
            BLOCKS = BLOCKS + 1
            padded = True

        # make placeholders
        dx, db, dc, ddelta = (torch.zeros_like(b).float().cuda() for b in [x, b, c, delta])
        da = torch.zeros(Ba, N, D, BLOCKS).float().cuda()
        y, dy = (torch.ones(Ba, 1, D, L).float().cuda() for _ in range(2))
        h, dh = (torch.zeros(2, 2, Ba, N, D, BLOCKS).float().cuda() for _ in range(2))

        # with padding

        mamba_tt[(BLOCKS, Ba)](
            x,
            dx,
            a,
            da,
            b,
            db,
            c,
            dc,
            delta,
            ddelta,
            h[0],
            dh[0],
            y,
            dy,
            h[0],
            dh[0],
            back=0,
            step=1,
            L=L,
            K=BLOCKSIZE,
            D_step=D,
            D=D,
            N=N,
        )
        reduce(h, False, Ba * N * D)
        reduce(dh, True, Ba * N * D)
        mamba_tt[(BLOCKS, Ba)](
            x,
            dx,
            a,
            da,
            b,
            db,
            c,
            dc,
            delta,
            ddelta,
            h[1],
            dh[1],
            y,
            dy,
            h[1],
            dh[1],
            back=0,
            step=2,
            L=L,
            K=BLOCKSIZE,
            D_step=D,
            D=D,
            N=N,
        )
        ctx.save_for_backward(a, b, c, delta, x, d)

        _, _, _, _, x, y = reshape_inputs(a, b, c, delta, x, y, rev=True)

        if d is not None:
            y = y + d * x
        return y  # , h, dh

    def backward(ctx, grad_output_y):
        # if grad_output_y is None:
        #    raise ValueError("grad_output_y is None!")
        a, b, c, delta, x, d = ctx.saved_tensors
        # x.retain_grad()

        Ba, _, D, L = x.shape
        _, N, _, _ = b.shape
        # we need to hardwire BLOCKSIZE since we need to know the number of Blocks
        BLOCKSIZE = 16
        BLOCKS = math.ceil(L / BLOCKSIZE)
        print("Sequence_length:", L)
        print("Number of Blcoks:", BLOCKS)
        if BLOCKS % 2 != 0:
            BLOCKS = BLOCKS + 1
            padded = True
        dx, da, db, dc, ddelta = (torch.zeros_like(b).float().cuda() for b in [x, a, b, c, delta])
        da = torch.zeros(Ba, N, D, L).float().cuda()
        y, dy = (torch.ones(Ba, 1, D, L).float().cuda() for _ in range(2))
        h, dh = (torch.zeros(2, 2, Ba, N, D, BLOCKS).float().cuda() for _ in range(2))
        # assert BLOCKS == SEQLEN // K
        mamba_tt[(BLOCKS, Ba)](
            x,
            dx,
            a,
            da,
            b,
            db,
            c,
            dc,
            delta,
            ddelta,
            h[0],
            dh[0],
            y,
            dy,
            h[0],
            dh[0],
            back=1,
            step=1,
            L=L,
            K=BLOCKSIZE,
            D_step=D,
            D=D,
            N=N,
        )
        reduce(h, False, Ba * N * D)
        reduce(dh, True, Ba * N * D)
        mamba_tt[(BLOCKS, Ba)](
            x,
            dx,
            a,
            da,
            b,
            db,
            c,
            dc,
            delta,
            ddelta,
            h[1],
            dh[1],
            y,
            dy,
            h[1],
            dh[1],
            back=1,
            step=2,
            L=L,
            K=BLOCKSIZE,
            D_step=D,
            D=D,
            N=N,
        )
        da = da.sum(-1, keepdim=True)

        if d is not None:
            dd = x
        else:
            dd = None
        da, db, dc, ddelta, dx, _ = reshape_inputs(da, db, dc, ddelta, x, y, rev=True)
        return dx, da, db, dc, ddelta, dd
