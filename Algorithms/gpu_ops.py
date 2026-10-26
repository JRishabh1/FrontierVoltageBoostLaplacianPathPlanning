"""Optional CUDA acceleration. No imports or device initialisation at import time.

CuPy supplies NVRTC-fused stencils and its exact Euclidean distance transform.
Without CuPy the public planners retain their PyTorch/SciPy implementations.
No approximate distance transform, reduced precision, or changed FVB threshold
is silently substituted. CUDA errors propagate rather than falling back.
"""
from functools import lru_cache

import numpy as np


def cupy_for(device):
    if str(device).split(':')[0] != 'cuda':
        return None
    try:
        import cupy as cp
    except ImportError:
        return None
    return cp


def distance_edt(mask, device='cpu'):
    """Exact float64 EDT returned as a torch tensor, plus implementation name.

2D/3D CUDA uses CuPy's parallel banding algorithm. Other dimensions and
CPU use SciPy's linear-time EDT, never an O(cells * surface) pairwise search.
"""
    import torch
    cp = cupy_for(device)
    if cp is not None and mask.ndim in (2, 3) and np.any(~mask):
        from cupyx.scipy.ndimage import distance_transform_edt
        # DLPack and the external stream keep ownership and ordering explicit.
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            d = distance_transform_edt(cp.asarray(mask), float64_distances=True)
            return torch.from_dlpack(d), 'cupy-exact-edt'
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(mask)
    return torch.as_tensor(d, dtype=torch.float64, device=device), 'scipy-exact-edt'


@lru_cache(maxsize=4)
def stencil_kernel(kind, single=False):
    import cupy as cp
    scalar = 'float' if single else 'double'
    if kind == 'laplace':
        body = '''
        if (!active[i]) { out[i] = in[i]; return; }
        int y = i / w, x = i % w;
        T up = in[y > 0 ? i-w : i], down = in[y+1 < h ? i+w : i];
        T left = in[x > 0 ? i-1 : i], right = in[x+1 < w ? i+1 : i];
        out[i] = ((up + down) + left + right) * T(0.25);
        '''
    elif kind == 'eikonal':
        body = '''
        if (!active[i]) { out[i] = in[i]; return; }
        int y = i / w, x = i % w;
        T a = fmin(y > 0 ? in[i-w] : T(INFINITY),
                   y+1 < h ? in[i+w] : T(INFINITY));
        T b = fmin(x > 0 ? in[i-1] : T(INFINITY),
                   x+1 < w ? in[i+1] : T(INFINITY));
        T lo = fmin(a, b), hi = fmax(a, b), s = inv[i];
        T t = lo + s;
        if (isfinite(hi) && hi-lo < s) {
            T d = hi-lo;
            // Difference formulation avoids catastrophic cancellation of T^2.
            t = lo + T(0.5) * (d + sqrt(fmax(T(0), T(2)*s*s-d*d)));
        }
        out[i] = fmin(in[i], t);
        '''
    else:
        raise ValueError(kind)
    # NVRTC predeclares device math intrinsics; no host/CUDA-toolkit headers
    # are needed (important on pip-only CUDA installations).
    src = f'''#define INFINITY (__int_as_float(0x7f800000))
    typedef {scalar} T;
    extern "C" __global__ void stencil(const T* in, T* out,
        const bool* active, const T* inv, int h, int w) {{
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= h*w) return;
        {body}
    }}'''
    return cp.RawKernel(src, 'stencil', options=('--std=c++11',))


def warmup_cuda_kernels():
    """Compile outside measured solve time (the worker records setup separately)."""
    if cupy_for('cuda') is not None:
        for kind, single in [('laplace', False), ('eikonal', False), ('eikonal', True)]:
            stencil_kernel(kind, single).compile()


def fvb_cuda(grid, start, end, obstacle, solved, free, n_l, epsilon):
    import cupy as cp
    kernel = stencil_kernel('laplace')
    phi = cp.ones(grid.shape, dtype=cp.float64)
    phi[end[1], end[0]] = 0.0
    phi[cp.asarray(obstacle)] = 1.0
    scratch = cp.empty_like(phi)
    obstacle, solved, free = (cp.asarray(a) for a in (obstacle, solved, free))
    launch = ((grid.size + 255) // 256,), (256,)
    shape = (np.int32(grid.shape[0]), np.int32(grid.shape[1]))
    vmax = 1.0
    while not bool(solved[start[1], start[0]].item()):
        for _ in range(n_l):
            kernel(*launch, (phi, scratch, free, phi, *shape))
            phi, scratch = scratch, phi
        new = free & (phi <= vmax - epsilon)
        solved |= new
        free &= ~new
        vmax += 1.0
        cp.copyto(phi, vmax, where=free | obstacle)
    return cp.asnumpy(phi)


def eikonal_cuda(initial, fixed, inv_speed, limit, tolerance, check_every, single=False):
    import cupy as cp
    dtype = cp.float32 if single else cp.float64
    field = cp.asarray(initial, dtype=dtype)
    scratch = cp.empty_like(field)
    active = cp.asarray(~fixed)
    inv = cp.asarray(inv_speed, dtype=dtype)
    kernel = stencil_kernel('eikonal', single)
    launch = ((field.size + 255) // 256,), (256,)
    shape = (np.int32(field.shape[0]), np.int32(field.shape[1]))
    converged = False
    for sweep in range(1, limit + 1):
        kernel(*launch, (field, scratch, active, inv, *shape))
        if sweep % check_every == 0 or sweep == limit:
            changed = ((scratch < field - tolerance)
                       | (cp.isfinite(scratch) & ~cp.isfinite(field)))
            converged = not bool(changed.any().item())
        field, scratch = scratch, field
        if converged:
            break
    return cp.asnumpy(field), sweep, converged
