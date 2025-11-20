# fed_utils/rank_allocator.py
import math
import numpy as np

def allocate_r_tot_for_client(layers, B_down_bytes, bytes_per_col):
    """
    下载阶段：对客户端 i 在总下行预算 B_down_bytes 下，
    以 单位字节收益 = sigma[j]^2 / bytes_per_col[layer] 的前缀增量水位法，得到 r_tot[layer]。
    layers: Dict[layer_name] -> {"sigma": np.ndarray (desc), "d_out":int, "d_in":int}
    bytes_per_col: Dict[layer_name] -> int  # (d_out+d_in)*bytes_down
    """
    r_tot = {L: 0 for L in layers}
    cost_used = 0
    # Normalise eigenvalues per layer so gains are compared layer-wise.
    norm_sigma = {}
    for L, meta in layers.items():
        sigma = meta.get("sigma", np.array([]))
        denom = float(np.sum(sigma)) if sigma is not None else 0.0
        norm_sigma[L] = (sigma / max(denom, 1e-12)).astype(float)

    while True:
        best = None
        for L, meta in layers.items():
            r = r_tot[L]
            if r < len(norm_sigma[L]):
                gain = norm_sigma[L][r] ** 2 / max(bytes_per_col[L], 1)
                if (best is None) or (gain > best[0]):
                    best = (gain, L)
        if best is None: break
        _, Lbest = best
        if cost_used + bytes_per_col[Lbest] > B_down_bytes: break
        r_tot[Lbest] += 1
        cost_used += bytes_per_col[Lbest]
    return r_tot, cost_used

def allocate_r_main_for_client(layers, r_tot, M_bytes, T_ms, c_mem_per_col, c_time_per_col, alpha=None, beta=None):
    """
    训练阶段：双约束（显存+时间）前缀增量，得到 r_main[layer]（<= r_tot[layer]）。
    c_mem_per_col: Dict[layer] -> bytes
    c_time_per_col: Dict[layer] -> ms
    """
    r_main = {L: 0 for L in layers}
    M_left, T_left = M_bytes, T_ms

    # Normalise eigenvalues per layer so we rank by relative (within-layer) importance.
    norm_sigma = {}
    for L, meta in layers.items():
        sigma = meta.get("sigma", np.array([]))
        denom = float(np.sum(sigma)) if sigma is not None else 0.0
        norm_sigma[L] = (sigma / max(denom, 1e-12)).astype(float)

    # Reserve a minimum per layer so r_tot - r_main <= 2 whenever budgets allow.
    for L, rt in r_tot.items():
        target_min = max(rt - 2, 0)
        capped_target = min(target_min, len(norm_sigma[L]))
        while r_main[L] < capped_target:
            if (M_left < c_mem_per_col[L]) or (T_left < c_time_per_col[L]):
                break
            r_main[L] += 1
            M_left -= c_mem_per_col[L]
            T_left -= c_time_per_col[L]

    while True:
        a = (1.0 / max(T_left, 1e-9)) if alpha is None else alpha
        b = (1.0 / max(M_left, 1e-9)) if beta  is None else beta
        best = None
        for L, meta in layers.items():
            r = r_main[L]
            if r >= min(r_tot[L], len(norm_sigma[L])):
                continue
            unit_cost = a * c_time_per_col[L] + b * c_mem_per_col[L]
            gain = norm_sigma[L][r] ** 2 / max(unit_cost, 1e-9)
            if (best is None) or (gain > best[0]):
                best = (gain, L)
        if best is None: break
        _, Lbest = best
        if (M_left < c_mem_per_col[Lbest]) or (T_left < c_time_per_col[Lbest]):
            break
        r_main[Lbest] += 1
        M_left -= c_mem_per_col[Lbest]
        T_left -= c_time_per_col[Lbest]

    # Ensure r_main is close to r_tot: r_tot - r_main <= 2 when resources allow.
    for L, rt in r_tot.items():
        target_min = max(rt - 2, 0)
        while r_main[L] < target_min:
            if (M_left < c_mem_per_col[L]) or (T_left < c_time_per_col[L]):
                break
            r_main[L] += 1
            M_left -= c_mem_per_col[L]
            T_left -= c_time_per_col[L]
    return r_main, (M_bytes - M_left), (T_ms - T_left)
