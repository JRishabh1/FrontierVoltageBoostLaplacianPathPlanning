import importlib.util
import math
import os
import sys

import numpy as np

from Algorithms import Algorithm2D


def _load_common():
    name = 'external_algorithms_common'
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'externalcommon.py')
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_common = _load_common()


def _sigma_p(x):
    """Eq. (7): x²(3 − 2x) for x ≤ 1, else 1."""
    return np.where(x <= 1.0, x * x * (3.0 - 2.0 * x), 1.0)


def _sigma_v(x):
    """Eq. (8): x² for x ≥ 0, else 0."""
    return np.where(x >= 0.0, x * x, 0.0)


def _xi_1(x, eps):
    """ξ1(x; ε) = 1 − σp(x / ε)."""
    return 1.0 - _sigma_p(x / eps)


def _xi_2(x, eps):
    """Smooth step from 0 (x < ε) to 1 (x > 1)."""
    t = np.clip((x - eps) / (1.0 - eps), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class HarmonicMapNavigation2D(Algorithm2D):
    """Harmonic-map navigation (Vlantis, Bechlioulis & Kyriakopoulos), 2D.

    Two stages, as in the paper (Sensors 2023, 23(9):4464; ICRA 2018):

    1. **Harmonic map** — maxchaos/hntf2d (C++, pinned in ``third_party/``,
       built with ``python third_party/setup_external.py hntf2d``) maps the
       free workspace onto the unit disk: the outer boundary goes to the unit
       circle and every obstacle collapses onto a distinct interior point
       ("puncture"). The domain is the face-connected free region of the
       start; its boundary is traced from the cell squares and split into
       panels of at most ``panel_length`` cells for the panel method.
    2. **Control law** — the paper's adaptive harmonic potential on the disk,
       pulled back through the map's Jacobian (eqs. 6–16; the controller is
       not part of the upstream repository, so it is implemented here):

           phi = k_d ln(|q − q_d|/2) − Σ_i k_i ln(|q − q_i|/2)
           psi = (1 + tanh(phi / w_phi)) / 2
           u   = −K_u s(q, k) J⁻¹(p) ∇psi

       with the adaptive laws for k_d (raised only near degenerate critical
       points) and k_i (decayed near the outer boundary so it stays
       repulsive). Defaults are the paper's Section 6.1 values.

    The closed loop is integrated with explicit Euler. The paper's time scale
    is arbitrary for path generation, so each step lasts as long as allowed by
    four limits: at most ``step_size`` cells of travel, at most half the
    workspace distance to the nearest wall, at most half the remaining disk
    distance to the unit circle or a puncture it is approaching (so the
    boundary gate s(q) — active only within ε_p of the circle — and the k_i
    decay take effect before the boundary is crossed), and at most
    ``dt_max``. Near walls the steps therefore shrink geometrically, as the
    continuous-time controller would slow down. The gains k are advanced over the same interval in
    sub-steps of at most ``dt`` (exact exponential update for the linear k_i
    law, so they stay within [0, k̄_i]).

    Maps where start and goal are not face-connected (only a diagonal pinch
    joins them) are outside the method's continuous-domain model and are
    rejected with ``UnsupportedMapError``.
    """

    def __init__(self, map, panel_length: float = 1.0, step_size: float = 1.0,
                 k_d: float = 20.0, k_i: float = 1.0, k_bar: float = 20.0,
                 w_phi: float = 20.0, K_u: float = 100.0, K_k: float = 100.0,
                 gamma: float = 0.7, alpha: float = 1.0, eps_p: float = 0.025,
                 eps_v: float = 0.1, eps_1: float = 0.01, eps_2: float = 0.1,
                 eps_3: float = 0.1, m: int = -2, dt: float = 0.01,
                 dt_max: float = 0.5, max_iters: int = 20000, stall_window: int = 2000):
        super().__init__(map)
        self.panel_length = panel_length
        self.step_size = step_size
        self.k_d0, self.k_i0, self.k_bar = k_d, k_i, k_bar
        self.w_phi, self.K_u, self.K_k = w_phi, K_u, K_k
        self.gamma, self.alpha = gamma, alpha
        self.eps_p, self.eps_v = eps_p, eps_v
        self.eps_1, self.eps_2, self.eps_3 = eps_1, eps_2, eps_3
        self.m = m
        self.dt, self.dt_max = dt, dt_max
        self.max_iters = max_iters
        self.stall_window = stall_window
        self._ext = _common.import_built('external_hntf2d', 'hntf2d')
        self._poly2circle = self._load_poly2circle()
        self._path = None
        self.hmap = None
        self._edges = None   # boundary segments (start points, end points)

    @staticmethod
    def _load_poly2circle():
        path = os.path.join(_common.source_dir('hntf2d', 'hntf2d'), 'cy', 'hntf2d', 'utils.py')
        spec = importlib.util.spec_from_file_location('hntf2d_upstream_utils', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.poly2circle

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid
        start, end = self.map.start, self.map.end
        domain = _common.free_component(grid, start)
        if not domain[end[1], end[0]]:
            raise _common.UnsupportedMapError(
                'start and goal are not face-connected — only a diagonal gap joins '
                'them, which has zero width in the harmonic map\'s continuous domain')

        self.hmap, loops = self._build_map(domain)
        self._edges = (np.vstack([l[:-1] for l in loops]), np.vstack([l[1:] for l in loops]))
        self._path = self._navigate(domain, start, end)
        if visualize:
            self._visualize()
        return self._path

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    # ------------------------------------------------------------------
    # Stage 1: harmonic map (hntf2d)
    # ------------------------------------------------------------------

    def _build_map(self, domain):
        loops = _common.domain_boundaries(domain)
        outer = [l for l in loops if _common.polygon_area(l) > 0]
        holes = [l for l in loops if _common.polygon_area(l) < 0]
        assert len(outer) == 1, 'a face-connected region has exactly one outer boundary'

        hm = self._ext.HarmonicMap2D()
        outer_panels = _common.resample_polygon(outer[0], self.panel_length)
        hm.boundary_append(outer_panels)
        for hole in holes:
            hm.boundary_append(_common.resample_polygon(hole, self.panel_length))
        uo, vo = self._poly2circle(outer_panels).T
        hm.solve(uo, vo)
        return hm, loops

    def _wall_distance(self, p):
        """Euclidean distance from p to the domain boundary polygons."""
        a, b = self._edges
        ab = b - a
        t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.einsum('ij,ij->i', ab, ab), 0.0, 1.0)
        return float(np.sqrt(np.min(np.sum((a + t[:, None] * ab - p) ** 2, axis=1))))

    # ------------------------------------------------------------------
    # Stage 2: adaptive harmonic potential + Jacobian pull-back
    # ------------------------------------------------------------------

    def _navigate(self, domain, start, end):
        hm = self.hmap
        punct = hm.puncts()
        P = np.zeros((0, 2)) if punct is None else np.asarray(punct, dtype=float).reshape(-1, 2)
        n = len(P)
        q_d = np.asarray(hm.map(*end), dtype=float)

        k_d = float(self.k_d0)
        k = np.full(n, float(self.k_i0))
        k_bar = np.full(n, float(self.k_bar))

        grid = domain.astype(np.uint8) ^ 1   # cells outside the domain are blocked
        p = np.array(start, dtype=float)
        path = [tuple(p)]
        anchor, since_anchor = p.copy(), 0

        for _ in range(self.max_iters):
            q = np.asarray(hm.map(*p), dtype=float)
            J = np.asarray(hm.jacob(*p), dtype=float)
            terms = self._terms(q, q_d, P, k_d, k, k_bar)
            if terms is None:
                return None
            try:
                u = -self.K_u * terms['s'] * np.linalg.solve(J, terms['grad_psi'])
            except np.linalg.LinAlgError:
                return None   # singular Jacobian (numerically degenerate map)
            speed = float(np.hypot(*u))
            if not math.isfinite(speed) or speed == 0.0:
                return None

            # Travel at most step_size, and at most half the distance to the
            # nearest wall: the panel-method map is only approximate within
            # about a panel of the boundary, so the disk-space limit alone
            # cannot keep a cell-scale step inside the domain.
            dt_eff = min(min(self.step_size, 0.5 * self._wall_distance(p)) / speed,
                         self.dt_max, self._boundary_dt(q, P, terms))
            k_d, k = self._advance_gains(q, q_d, P, k_d, k, k_bar, dt_eff, terms)

            p = p + dt_eff * u
            if not _common.in_free_cell(grid, (p[1], p[0])):
                return None
            path.append((float(p[0]), float(p[1])))

            if abs(end[0] - p[0]) <= 1 and abs(end[1] - p[1]) <= 1:
                path.append((float(end[0]), float(end[1])))
                return path

            # Stuck at a saddle (or crawling against the outer boundary).
            if np.hypot(*(p - anchor)) >= 1.0:
                anchor, since_anchor = p.copy(), 0
            else:
                since_anchor += 1
                if since_anchor >= self.stall_window:
                    return None
        return None

    def _boundary_dt(self, q, P, t):
        """Longest step that closes at most half the disk distance to the unit
        circle or to any puncture it is moving towards.

        In the disk the motion is q̇ = J u = −K_u s ∇ψ. The boundary gate s is
        only active within ε_p of the circle — far thinner than a workspace
        cell — so without this limit a single step could jump across it before
        s and the k_i decay act. Motion away from a boundary is not limited.
        """
        q_dot = -self.K_u * t['s'] * t['grad_psi']
        limit = math.inf
        closing = float(q_dot @ q) / t['nq']
        if closing > 0:
            limit = 0.5 * max(1.0 - t['nq'], 0.0) / closing
        if len(P):
            dist = np.sqrt(t['r_i2'])
            closing_i = -(t['dQ'] @ q_dot) / dist
            towards = closing_i > 0
            if towards.any():
                limit = min(limit, float(np.min(0.5 * dist[towards] / closing_i[towards])))
        return limit

    def _terms(self, q, q_d, P, k_d, k, k_bar):
        """Potential, gain function s and adaptive-law ingredients at q."""
        dq_d = q - q_d
        r_d2 = float(dq_d @ dq_d)
        dQ = q - P
        r_i2 = np.einsum('ij,ij->i', dQ, dQ)
        if r_d2 == 0.0 or np.any(r_i2 == 0.0):
            return None

        # (9)-(11): ln(|x|/2) = ½ ln(|x|² / 4);  ∇ ln|x| = x / |x|²
        phi = 0.5 * k_d * math.log(r_d2 / 4.0) - 0.5 * float(k @ np.log(r_i2 / 4.0))
        grad_phi = k_d * dq_d / r_d2 - (k / r_i2) @ dQ
        th = math.tanh(phi / self.w_phi)
        grad_psi = (1.0 - th * th) / (2.0 * self.w_phi) * grad_phi

        # (7): s vanishes at the outer boundary / when moving away from the centre
        nq = float(np.hypot(*q))
        ng = float(np.hypot(*grad_psi))
        s = (self.gamma * float(_sigma_p((1.0 - nq) / self.eps_p))
             + (1.0 - self.gamma) * float(_sigma_v((grad_psi @ q) / (self.eps_v + ng * nq))))

        return {'phi': phi, 'grad_phi': grad_phi, 'grad_psi': grad_psi, 'th': th,
                's': s, 'nq': nq, 'ng': ng, 'dq_d': dq_d, 'r_d2': r_d2, 'dQ': dQ, 'r_i2': r_i2}

    def _gain_rates(self, q, q_d, P, k_d, k, k_bar, t):
        """Eq. (12): returns (dk_d/dt, A, B) with dk_i/dt = A_i − B_i k_i."""
        dq_d, r_d2, dQ, r_i2 = t['dq_d'], t['r_d2'], t['dQ'], t['r_i2']
        grad_phi, grad_psi, s, nq, ng = t['grad_phi'], t['grad_psi'], t['s'], t['nq'], t['ng']

        # λ: non-negative eigenvalue of ∇²φ (traceless: eigenvalues ±λ).
        # ∇² ln|x| = (I |x|² − 2 x xᵀ) / |x|⁴
        H = k_d * (np.eye(2) * r_d2 - 2.0 * np.outer(dq_d, dq_d)) / (r_d2 * r_d2)
        H -= np.eye(2) * float(np.sum(k / r_i2))
        H += 2.0 * np.einsum('i,ij,ik->jk', k / (r_i2 * r_i2), dQ, dQ)
        lam = math.hypot(H[0, 0], H[0, 1])
        kd_rate = float(_xi_1(lam + math.hypot(*grad_phi), self.eps_1))

        if len(P) == 0:
            return kd_rate, np.zeros(0), np.zeros(0)

        # (13)-(14)
        ell = -self.K_u * s / np.log(np.sqrt(r_i2) / 2.0)
        g0 = float(_sigma_v(0.25 * (self.alpha * ng * math.sqrt(r_d2) - grad_psi @ dq_d)))
        gi = _sigma_v(0.5 * (dQ @ grad_psi))

        # (15)-(16): weights from r_0 = (1 − |q|)², r_i = |q − q_i|²
        # r̄_i = (Σ_{j≠i} r_j^m)^(1/m), a smooth under-approximation of min_{j≠i} r_j
        r = np.concatenate([[(1.0 - nq) ** 2], r_i2])
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            rm = r ** self.m
            others = np.where(np.eye(len(r), dtype=bool), 0.0, rm[None, :]).sum(axis=1)
            r_bar = others ** (1.0 / self.m)
            w_bar = r_bar / (r + r_bar)
        w_bar = np.nan_to_num(w_bar, nan=1.0)
        denom = w_bar[0] + float(k_bar @ w_bar[1:])
        w0 = float(_xi_2(w_bar[0], self.eps_3)) / denom
        wi = w_bar[1:] / denom

        # h_i: speed up the decay of the k_i pushing towards the outer boundary
        e_d = -dq_d / r_d2
        e_i = -dQ / r_i2[:, None]
        h_bar = k * (1.0 - t['th'] ** 2) ** 2 * (e_i @ e_d)
        sv = _sigma_v(h_bar)
        h = 1.0 + sv / (1.0 + sv.sum())

        a = wi * ell * gi
        A = k_bar * a
        B = a + self.K_k * h * w0 * (g0 + float(_xi_1(s, self.eps_2)))
        return kd_rate, A, B

    def _advance_gains(self, q, q_d, P, k_d, k, k_bar, dt_eff, terms):
        n_sub = max(1, math.ceil(dt_eff / self.dt))
        h = dt_eff / n_sub
        for i in range(n_sub):
            if i:
                terms = self._terms(q, q_d, P, k_d, k, k_bar)
            kd_rate, A, B = self._gain_rates(q, q_d, P, k_d, k, k_bar, terms)
            k_d += h * kd_rate
            if len(k):
                decay = np.exp(-B * h)
                with np.errstate(divide='ignore', invalid='ignore'):
                    k_star = np.where(B > 0, A / B, k)
                k = k_star + (k - k_star) * decay
        return k_d, k

    def _visualize(self):
        from PIL import Image
        grey = self.map.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()
        for x, y in self._path or []:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < rgb.shape[0] and 0 <= xi < rgb.shape[1]:
                rgb[yi, xi] = [128, 128, 255]
        sx, sy = self.map.start
        ex, ey = self.map.end
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]
        Image.fromarray(rgb, mode='RGB').save('harmonic_map_result.png')
