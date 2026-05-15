#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abs_measured_final.py

用途
====
用于论文“基于格的多授权机构动态属性签名方案”的实验部分。

本代码采用“基础操作实测 + 表3结构公式代入”的方法：
1. 先实际测量基础操作耗时：
   T_mul_Zq, T_ntt_Rq, T_sample, T_hash, T_PRF, T_part, T_proxy, T_rev, T_cmp
2. 再根据修正后的表3计算各方案在不同属性数量下的签名生成、验证、KeyGen墙钟时间和签名尺寸。
3. 输出 CSV、Overleaf 表格和可选图片。

重要边界
========
本代码不声称完整复现所有参考文献方案。对于未公开完整代码和参数的文献，本文采用基础操作实测
与理论结构项代入的归一化实验方法。代码中的 T_PRF、T_part、T_proxy、T_rev、T_cmp 分别是对应
结构项的可测量抽象操作，用于反映文献方案中确实存在的额外功能结构，而不是原文逐行实现。

默认参数参考 ML-DSA-44 / Dilithium2 风格：
q = 8380417, n = 256, R_q = Z_q[X]/(X^256+1), k = 4。

运行示例
========
快速运行：
    python abs_measured_final.py --attrs 4,8,16,32 --repeat 5 --zq-dim 256 --out results_fast

论文建议：
    python abs_measured_final.py --attrs 4,8,16,32,64,128,256 --repeat 10 --zq-dim 1024 --aa-count 4 --out results_paper

说明：
    默认不使用 numpy 加速 Z_q 乘法，以避免“Z_q 使用底层 BLAS、R_q/NTT 使用纯 Python”导致比较不公平。
    若仅想观察本机优化库效果，可额外加入 --use-numpy。

如果 CSV 被 Excel/WPS 占用，代码会自动另存为带时间戳的文件名。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

# 为保证论文实验中 Z_q 普通乘法与 R_q/NTT 均在同一 Python 实现层级下比较，
# 默认不使用 numpy 加速 Z_q 乘法。若用户希望观察本机 BLAS 加速效果，可显式加 --use-numpy。
USE_NUMPY = False


# =============================================================================
# 1. 参数与方案画像
# =============================================================================

@dataclass(frozen=True)
class Params:
    """统一实验参数。"""

    q: int = 8380417
    n_poly: int = 256
    k_ref: int = 4
    eta: int = 2
    gamma1: int = 2 ** 17
    beta_mldsa: int = 78
    zq_dim: int = 1024
    aa_count: int = 4
    revocation_list_size: int = 256
    cmp_vector_len: int = 256

    @property
    def ell_q(self) -> int:
        return math.ceil(math.log2(self.q))

    @property
    def equivalent_zq_dim(self) -> int:
        return self.k_ref * self.n_poly

    @property
    def norm_bound(self) -> int:
        return self.gamma1 - self.beta_mldsa


@dataclass(frozen=True)
class Timings:
    """实测基础操作耗时，单位 ms。"""

    T_mul_Zq: float
    T_ntt_Rq: float
    T_sample: float
    T_hash: float
    T_PRF: float
    T_part: float
    T_proxy: float
    T_rev: float
    T_cmp: float


@dataclass(frozen=True)
class Scheme:
    """对比方案结构画像。

    sign_expr / verify_expr 对应表3中的符号结构。
    """

    name: str
    citation_key: str
    domain: str                  # "Zq" or "Rq"
    authority: str               # "single" or "multi"
    constant_signature: bool
    sign_expr: str
    verify_expr: str
    size_expr: str


def build_schemes() -> List[Scheme]:
    """根据修正后的表3构造方案列表。

    注：这里不写任何人为固定耗时。所有公式中的基础项都来自 measure_all_timings() 的实测结果。
    """
    return [
        Scheme(
            name="Luo et al. [32]",
            citation_key="luo2021abs",
            domain="Zq",
            authority="single",
            constant_signature=False,
            sign_expr="T_eval_Zq + T_proxy + T_sample + T_hash",
            verify_expr="T_eval_Zq + T_proxy + T_hash",
            size_expr="O(ell)",
        ),
        Scheme(
            name="Luo et al. [35]",
            citation_key="luo2022unbounded",
            domain="Zq",
            authority="single",
            constant_signature=False,
            sign_expr="T_eval_Zq + N_attr*T_PRF + T_part + T_sample + T_hash",
            verify_expr="T_eval_Zq + N_attr*T_PRF + T_part + T_hash",
            size_expr="O(ell*poly(n))",
        ),
        Scheme(
            name="Liu et al. [39]",
            citation_key="liu2024revocable",
            domain="Zq",
            authority="single",
            constant_signature=False,
            sign_expr="T_eval_Zq + T_rev + T_sample + T_hash",
            verify_expr="T_eval_Zq + T_cmp + T_hash",
            size_expr="O(ell)",
        ),
        Scheme(
            name="Kong et al. [40]",
            citation_key="kong2024flexible",
            domain="Zq",
            authority="single",
            constant_signature=True,
            sign_expr="T_eval_Zq + T_sample + T_hash",
            verify_expr="T_eval_Zq + T_hash",
            size_expr="O(1)",
        ),
        Scheme(
            name="Luo et al. [38]",
            citation_key="luo2026constant",
            domain="Zq",
            authority="single",
            constant_signature=True,
            sign_expr="T_eval_Zq + T_sample + T_hash",
            verify_expr="T_eval_Zq + T_hash",
            size_expr="O(1)",
        ),
        Scheme(
            name="Proposed scheme",
            citation_key="ours",
            domain="Rq",
            authority="multi",
            constant_signature=True,
            sign_expr="T_eval_Rq + T_sample + T_hash",
            verify_expr="T_eval_Rq + T_hash",
            size_expr="O(1)",
        ),
    ]


# =============================================================================
# 2. 安全写文件
# =============================================================================

def make_out_dir(base: str) -> Path:
    """创建输出目录。如果目录已存在，则自动追加时间戳，避免覆盖和权限冲突。"""
    p = Path(base)
    if p.exists():
        p = Path(f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """安全写 CSV。如果文件被 Excel/WPS 占用，自动改写到带时间戳的新文件。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    for i in range(5):
        try:
            with open(target, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            return
        except PermissionError:
            target = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}{path.suffix}")
    raise PermissionError(f"无法写入 {path}，请关闭 Excel/WPS 或更换 --out 目录。")


def safe_write_text(path: Path, text: str) -> None:
    """安全写文本文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    for i in range(5):
        try:
            target.write_text(text, encoding="utf-8")
            return
        except PermissionError:
            target = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}{path.suffix}")
    raise PermissionError(f"无法写入 {path}，请关闭相关文件或更换 --out 目录。")


# =============================================================================
# 3. 基础运算：Z_q 矩阵乘法、R_q NTT、多项式运算
# =============================================================================

def factor_distinct(n: int) -> List[int]:
    """返回 n 的不同素因子。"""
    factors: List[int] = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def primitive_root_mod_prime(q: int) -> int:
    """寻找素数 q 的一个原根。对 Dilithium 模数直接使用 10。"""
    if q == 8380417:
        return 10
    phi = q - 1
    factors = factor_distinct(phi)
    for g in range(2, q):
        ok = True
        for p in factors:
            if pow(g, phi // p, q) == 1:
                ok = False
                break
        if ok:
            return g
    raise ValueError("未找到原根，请检查 q 是否为素数。")


def bit_reverse_copy(a: List[int]) -> List[int]:
    """NTT 的 bit-reversal 重排。"""
    n = len(a)
    bits = n.bit_length() - 1
    out = [0] * n
    for i, x in enumerate(a):
        j = int(f"{i:0{bits}b}"[::-1], 2)
        out[j] = x
    return out


def ntt(a: List[int], q: int, root: int, invert: bool = False) -> List[int]:
    """迭代 NTT / inverse NTT。"""
    n = len(a)
    a = bit_reverse_copy([x % q for x in a])
    if invert:
        root = pow(root, q - 2, q)

    length = 2
    while length <= n:
        wlen = pow(root, n // length, q)
        half = length // 2
        for i in range(0, n, length):
            w = 1
            for j in range(i, i + half):
                u = a[j]
                v = a[j + half] * w % q
                a[j] = (u + v) % q
                a[j + half] = (u - v) % q
                w = w * wlen % q
        length *= 2

    if invert:
        inv_n = pow(n, q - 2, q)
        a = [x * inv_n % q for x in a]
    return a


class NTTContext:
    """R_q = Z_q[X]/(X^n+1) 上的 NTT 乘法上下文。"""

    def __init__(self, n: int, q: int):
        self.n = n
        self.q = q
        self.length = 2 * n
        if (q - 1) % self.length != 0:
            raise ValueError(f"q-1 必须能被 2n 整除。当前 q={q}, n={n}")
        primitive = primitive_root_mod_prime(q)
        self.root = pow(primitive, (q - 1) // self.length, q)

        # 简单校验 2n 阶单位根。
        if pow(self.root, self.length, q) != 1 or pow(self.root, self.n, q) == 1:
            raise ValueError("NTT root 校验失败。")

    def mul(self, a: List[int], b: List[int]) -> List[int]:
        """使用 2n 点 NTT 做普通卷积，再按 X^n=-1 折叠。"""
        n = self.n
        aa = a + [0] * n
        bb = b + [0] * n
        A = ntt(aa, self.q, self.root, invert=False)
        B = ntt(bb, self.q, self.root, invert=False)
        C = [(x * y) % self.q for x, y in zip(A, B)]
        c = ntt(C, self.q, self.root, invert=True)
        return [(c[i] - c[i + n]) % self.q for i in range(n)]


def build_zq_input(dim: int, q: int, seed: int):
    """生成 Z_q 矩阵-向量乘输入。优先使用 numpy，没有则回退纯 Python。"""
    if USE_NUMPY and np is not None:
        rng = np.random.default_rng(seed)
        A = rng.integers(0, q, size=(dim, dim), dtype=np.int64)
        x = rng.integers(0, q, size=(dim,), dtype=np.int64)
        return A, x

    rng = random.Random(seed)
    A = [[rng.randrange(q) for _ in range(dim)] for _ in range(dim)]
    x = [rng.randrange(q) for _ in range(dim)]
    return A, x


def zq_matvec_run(A, x, q: int) -> None:
    """执行一次 Z_q 矩阵-向量乘。"""
    if USE_NUMPY and np is not None:
        y = (A @ x) % q
        if int(y[0]) == -1:
            print("impossible")
        return

    y0 = 0
    for row in A:
        acc = 0
        for a, b in zip(row, x):
            acc += a * b
        y0 ^= acc % q
    if y0 == -1:
        print("impossible")


def build_poly_input(n: int, q: int, seed: int) -> Tuple[List[int], List[int]]:
    """生成 R_q 多项式乘法输入。"""
    rng = random.Random(seed)
    return [rng.randrange(q) for _ in range(n)], [rng.randrange(q) for _ in range(n)]


# =============================================================================
# 4. 基础结构项的可测量抽象
# =============================================================================

def sample_run(params: Params, rng: random.Random) -> None:
    """模拟一次短向量/离散高斯类采样。"""
    _ = [rng.randint(-params.eta, params.eta) for _ in range(params.equivalent_zq_dim)]


def hash_run(params: Params) -> None:
    """模拟一次 SHAKE256 消息映射。"""
    out_bits = params.k_ref * params.n_poly * params.ell_q
    out_bytes = math.ceil(out_bits / 8)
    hashlib.shake_256(b"benchmark-message").digest(out_bytes)


def prf_run(params: Params, counter: int) -> None:
    """模拟一次 PRF.Eval。用于 Luo[35] 无界属性方案中的 PRF 矩阵生成项。

    这里用 SHAKE256(seed || counter) 输出一个短伪随机串作为 PRF 抽象。
    """
    data = b"prf-seed" + counter.to_bytes(8, "little", signed=False)
    hashlib.shake_256(data).digest(32)


def partition_run(params: Params) -> None:
    """模拟 partitioning function 及其兼容算法的基础开销。

    该抽象对应 Luo[35] 中 partitioning function、Encode/PubEval/TrapEval 相关消息处理。
    """
    digest = hashlib.shake_256(b"partitioning-function-input").digest(128)
    acc = 0
    for b in digest:
        acc ^= (b * 1315423911) & 0xFFFFFFFF
    if acc == -1:
        print("impossible")


def proxy_run(params: Params, A_proxy, x_proxy) -> None:
    """模拟代理重签名结构项 T_proxy。

    用一次较小维度的 Z_q 线性变换作为 proxy 结构项的可测量抽象。
    """
    zq_matvec_run(A_proxy, x_proxy, params.q)


def revocation_run(params: Params, rev_set: set, rng: random.Random) -> None:
    """模拟撤销列表检查 T_rev。"""
    # 做若干次 membership/hash 操作，反映撤销机制的基础查询开销。
    hit = 0
    for _ in range(16):
        user_id = rng.randrange(params.revocation_list_size * 4)
        token = hashlib.shake_256(str(user_id).encode()).digest(8)
        hit ^= (token in rev_set)
    if hit == -1:
        print("impossible")


def comparison_run(params: Params, a: List[int], b: List[int]) -> None:
    """模拟属性范围比较 / comparable 检查 T_cmp。"""
    acc = 0
    for x, y in zip(a, b):
        acc += 1 if x <= y else 0
    if acc == -1:
        print("impossible")


# =============================================================================
# 5. 计时函数
# =============================================================================

def time_call(fn: Callable[[], None], repeat: int, warmup: int = 2) -> Tuple[float, float]:
    """重复执行并返回平均值、标准差，单位 ms。"""
    for _ in range(max(0, warmup)):
        fn()

    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)

    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, std


def measure_all_timings(params: Params, repeat: int, seed: int) -> Tuple[Timings, List[Dict[str, object]]]:
    """实际测量所有基础操作。"""
    rng = random.Random(seed)

    # Z_q 乘法输入
    A_zq, x_zq = build_zq_input(params.zq_dim, params.q, seed)

    # R_q / NTT 输入
    ctx = NTTContext(params.n_poly, params.q)
    a_poly, b_poly = build_poly_input(params.n_poly, params.q, seed + 1)

    # proxy 使用较小但可测的 Z_q 变换，避免过度放大 proxy 项。
    proxy_dim = max(32, min(params.zq_dim // 4, 256))
    A_proxy, x_proxy = build_zq_input(proxy_dim, params.q, seed + 2)

    # rev/cmp 输入
    rev_set = {hashlib.shake_256(str(i).encode()).digest(8) for i in range(params.revocation_list_size)}
    cmp_a = [rng.randrange(params.q) for _ in range(params.cmp_vector_len)]
    cmp_b = [rng.randrange(params.q) for _ in range(params.cmp_vector_len)]

    results: List[Dict[str, object]] = []

    def record(name: str, fn: Callable[[], None]) -> float:
        mean, std = time_call(fn, repeat=repeat)
        results.append({"operation": name, "mean_ms": mean, "std_ms": std})
        return mean

    T_mul_Zq = record("T_mul_Zq", lambda: zq_matvec_run(A_zq, x_zq, params.q))
    T_ntt_Rq = record("T_ntt_Rq", lambda: ctx.mul(a_poly, b_poly))
    T_sample = record("T_sample", lambda: sample_run(params, rng))
    T_hash = record("T_hash", lambda: hash_run(params))
    T_PRF = record("T_PRF", lambda: prf_run(params, rng.randrange(1 << 30)))
    T_part = record("T_part", lambda: partition_run(params))
    T_proxy = record("T_proxy", lambda: proxy_run(params, A_proxy, x_proxy))
    T_rev = record("T_rev", lambda: revocation_run(params, rev_set, rng))
    T_cmp = record("T_cmp", lambda: comparison_run(params, cmp_a, cmp_b))

    timings = Timings(
        T_mul_Zq=T_mul_Zq,
        T_ntt_Rq=T_ntt_Rq,
        T_sample=T_sample,
        T_hash=T_hash,
        T_PRF=T_PRF,
        T_part=T_part,
        T_proxy=T_proxy,
        T_rev=T_rev,
        T_cmp=T_cmp,
    )
    return timings, results


# =============================================================================
# 6. 表3公式代入
# =============================================================================

def eval_zq(attrs: int, timings: Timings) -> float:
    """T_eval^{Z_q}(f) = O(N_attr * T_mul_Zq)。"""
    return attrs * timings.T_mul_Zq


def eval_rq(attrs: int, timings: Timings) -> float:
    """T_eval^{R_q}(f) = O(N_attr * T_ntt_Rq)。"""
    return attrs * timings.T_ntt_Rq


def estimate_sign(scheme: Scheme, attrs: int, timings: Timings) -> float:
    """根据表3估算签名生成耗时。"""
    if scheme.citation_key == "luo2021abs":
        return eval_zq(attrs, timings) + timings.T_proxy + timings.T_sample + timings.T_hash
    if scheme.citation_key == "luo2022unbounded":
        return eval_zq(attrs, timings) + attrs * timings.T_PRF + timings.T_part + timings.T_sample + timings.T_hash
    if scheme.citation_key == "liu2024revocable":
        return eval_zq(attrs, timings) + timings.T_rev + timings.T_sample + timings.T_hash
    if scheme.citation_key == "kong2024flexible":
        return eval_zq(attrs, timings) + timings.T_sample + timings.T_hash
    if scheme.citation_key == "luo2026constant":
        return eval_zq(attrs, timings) + timings.T_sample + timings.T_hash
    if scheme.citation_key == "ours":
        return eval_rq(attrs, timings) + timings.T_sample + timings.T_hash
    raise ValueError(f"Unknown scheme: {scheme.name}")


def estimate_verify(scheme: Scheme, attrs: int, timings: Timings) -> float:
    """根据表3估算验证耗时。"""
    if scheme.citation_key == "luo2021abs":
        return eval_zq(attrs, timings) + timings.T_proxy + timings.T_hash
    if scheme.citation_key == "luo2022unbounded":
        return eval_zq(attrs, timings) + attrs * timings.T_PRF + timings.T_part + timings.T_hash
    if scheme.citation_key == "liu2024revocable":
        return eval_zq(attrs, timings) + timings.T_cmp + timings.T_hash
    if scheme.citation_key == "kong2024flexible":
        return eval_zq(attrs, timings) + timings.T_hash
    if scheme.citation_key == "luo2026constant":
        return eval_zq(attrs, timings) + timings.T_hash
    if scheme.citation_key == "ours":
        return eval_rq(attrs, timings) + timings.T_hash
    raise ValueError(f"Unknown scheme: {scheme.name}")


def estimate_keygen(scheme: Scheme, attrs: int, timings: Timings, params: Params) -> Tuple[float, float]:
    """估算 KeyGen 总计算量和多授权并行墙钟时间。"""
    if scheme.domain == "Rq":
        per_attr = timings.T_ntt_Rq + timings.T_sample
    else:
        per_attr = timings.T_mul_Zq + timings.T_sample

    # 对于具有额外结构的方案，KeyGen 中也可能包含相应结构项。
    if scheme.citation_key == "luo2022unbounded":
        per_attr += timings.T_PRF
    elif scheme.citation_key == "liu2024revocable":
        per_attr += timings.T_rev
    elif scheme.citation_key == "luo2021abs":
        per_attr += timings.T_proxy

    total = attrs * per_attr
    if scheme.authority == "multi":
        wall = math.ceil(attrs / max(1, params.aa_count)) * per_attr
    else:
        wall = total
    return total, wall


def estimate_signature_size(scheme: Scheme, attrs: int, params: Params) -> int:
    """估算签名尺寸，仅用于趋势图。"""
    coeff_bits = math.ceil(math.log2(2 * params.norm_bound + 1))
    coeff_bytes = math.ceil(coeff_bits / 8)

    # 常数签名：参考 2*k_ref 个 R_q 多项式。
    const_size = 2 * params.k_ref * params.n_poly * coeff_bytes

    if scheme.constant_signature:
        return const_size

    if scheme.citation_key == "luo2022unbounded":
        # unbounded attributes 方案签名随属性长度/属性数量增长，且包含额外 poly(n) 项；
        # 这里用 1.5 倍线性估算体现其比普通线性方案更重的签名趋势。
        return int(1.5 * attrs * params.k_ref * params.n_poly * coeff_bytes)

    return attrs * params.k_ref * params.n_poly * coeff_bytes


# =============================================================================
# 7. Overleaf 表格
# =============================================================================

def table3_overleaf() -> str:
    """修正后的表3 Overleaf 版本。"""
    return r"""\begin{table}
	\centering
	\caption{核心后量子格基属性签名方案签名与验证开销对比}{Comparison of Signing and Verification Costs}
	\label{tab:cost_compare}
	\tabulinesep=1.5mm
	\begin{tabu}to \linewidth{X[c,m]X[c,m]X[c,m]X[c,m]}
		\tabucline[0.08em]-
		\textbf{方案} & \textbf{签名生成开销} & \textbf{签名验证开销} & \textbf{签名尺寸} \\
		\tabucline-

		Luo等\cite{luo2021abs}
		& $T^{\mathbb{Z}_q}_{eval}(f)+T_{proxy}+T_{sample}+T_{hash}$
		& $T^{\mathbb{Z}_q}_{eval}(f)+T_{proxy}+T_{hash}$
		& $O(\ell)$ \\

		Luo等\cite{luo2022unbounded}
		& $T^{\mathbb{Z}_q}_{eval}(f)+T_{PRF}(\ell)+T_{part}+T_{sample}+T_{hash}$
		& $T^{\mathbb{Z}_q}_{eval}(f)+T_{PRF}(\ell)+T_{part}+T_{hash}$
		& $O(\ell\cdot \mathrm{poly}(n))$ \\

		Liu等\cite{liu2024revocable}
		& $T^{\mathbb{Z}_q}_{eval}(f)+T_{rev}+T_{sample}+T_{hash}$
		& $T^{\mathbb{Z}_q}_{eval}(f)+T_{cmp}+T_{hash}$
		& $O(\ell)$ \\

		Kong等\cite{kong2024flexible}
		& $T^{\mathbb{Z}_q}_{eval}(f)+O(1)T_{sample}+T_{hash}$
		& $T^{\mathbb{Z}_q}_{eval}(f)+T_{hash}$
		& $O(1)$ \\

		Luo等\cite{luo2026constant}
		& $T^{\mathbb{Z}_q}_{eval}(f)+T_{sample}+T_{hash}$
		& $T^{\mathbb{Z}_q}_{eval}(f)+T_{hash}$
		& \textbf{$O(1)$} \\

		\midrule

		本文方案
		& $T^{R_q}_{eval}(f)+T_{sample}+T_{hash}$
		& $T^{R_q}_{eval}(f)+T_{hash}$
		& \textbf{$O(1)$} \\

		\tabucline[0.08em]-
	\end{tabu}
\end{table}
"""


def table3_note_overleaf() -> str:
    """表3注释。"""
    return r"""
其中，$T^{\mathbb{Z}_q}_{eval}(f)$ 表示在 $\mathbb{Z}_q$ 上执行访问策略求值或同态求值的主导开销，
$T^{R_q}_{eval}(f)$ 表示在多项式环 $R_q$ 上执行访问策略求值的主导开销。
若访问结构规模与属性数量线性相关，则有
$T^{\mathbb{Z}_q}_{eval}(f)=O(N_{\mathrm{attr}}T^{\mathbb{Z}_q}_{mul})$，
而本文方案可利用NTT加速，故有
$T^{R_q}_{eval}(f)=O(N_{\mathrm{attr}}T^{R_q}_{ntt})$。
$T_{PRF}(\ell)$ 表示无界属性方案中随属性长度增长的PRF矩阵生成相关开销，
$T_{part}$ 表示partitioning函数及其兼容算法相关开销，
$T_{proxy}$ 表示代理重签名结构相关开销，
$T_{rev}$ 与 $T_{cmp}$ 分别表示撤销和比较功能带来的额外开销。
"""


def params_table_overleaf(params: Params, schemes: Sequence[Scheme]) -> str:
    """实验参数表。"""
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{128-bit安全目标下的对比实验参数设置}{Parameters for comparison experiments under 128-bit security target}",
        r"\label{tab:params_compare}",
        r"\tabulinesep=1.2mm",
        r"\begin{tabu}to \linewidth{X[c,m]X[c,m]X[c,m]X[c,m]X[c,m]X[c,m]}",
        r"\tabucline[0.08em]-",
        r"\textbf{方案} & \textbf{计算域} & \textbf{模数$q$} & \textbf{维度设置} & \textbf{签名尺寸} & \textbf{授权模式} \\",
        r"\tabucline-",
    ]
    for s in schemes:
        if s.domain == "Rq":
            dim = rf"$R_q=\mathbb{{Z}}_q[X]/(X^{{{params.n_poly}}}+1)$"
        else:
            dim = rf"$N={params.zq_dim}$"
        sig = r"$O(1)$" if s.constant_signature else r"$O(N_{\mathrm{attr}})$"
        lines.append(f"{s.name} & ${s.domain}$ & ${params.q}$ & {dim} & {sig} & {s.authority} " + r"\\")
    lines += [r"\tabucline[0.08em]-", r"\end{tabu}", r"\end{table}"]
    return "\n".join(lines)


# =============================================================================
# 8. 绘图
# =============================================================================

def maybe_plot(out_dir: Path, bench_rows: List[Dict[str, object]], size_rows: List[Dict[str, object]]) -> None:
    """如果安装 matplotlib，则输出适合黑白打印的折线图。

    设计原则：
    1. 不依赖颜色区分曲线；
    2. 每条曲线使用不同线型和不同数据标记点；
    3. 默认全部使用黑色线条，保证黑白打印或灰度复印时仍可区分；
    4. 标记点使用白色填充和黑色边框，避免在打印中糊成一团。
    """
    if plt is None:
        return

    line_styles = [
        "-", "--", "-.", ":",
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
        (0, (5, 2, 1, 2)),
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    def style_for_index(idx: int):
        return line_styles[idx % len(line_styles)], markers[idx % len(markers)]

    def plot_metric(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(9, 5.5))
        names = list(dict.fromkeys(str(r["scheme"]) for r in bench_rows))

        for idx, name in enumerate(names):
            sub = [r for r in bench_rows if r["scheme"] == name]
            sub.sort(key=lambda x: int(x["attrs"]))
            xs = [int(r["attrs"]) for r in sub]
            ys = [float(r[metric]) for r in sub]
            linestyle, marker = style_for_index(idx)

            plt.plot(
                xs,
                ys,
                color="black",
                linestyle=linestyle,
                marker=marker,
                linewidth=1.6,
                markersize=5.5,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.0,
                label=name,
            )

        plt.xlabel("Number of attributes")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
        plt.legend(fontsize=8, frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    plot_metric("sign_ms", "Signing time (ms)", "sign_time_ms.png")
    plot_metric("verify_ms", "Verification time (ms)", "verify_time_ms.png")
    plot_metric("keygen_wall_ms", "KeyGen wall-clock time (ms)", "keygen_wall_time_ms.png")

    plt.figure(figsize=(9, 5.5))
    names = list(dict.fromkeys(str(r["scheme"]) for r in size_rows))

    for idx, name in enumerate(names):
        sub = [r for r in size_rows if r["scheme"] == name]
        sub.sort(key=lambda x: int(x["attrs"]))
        xs = [int(r["attrs"]) for r in sub]
        ys = [int(r["signature_size_bytes"]) for r in sub]
        linestyle, marker = style_for_index(idx)

        plt.plot(
            xs,
            ys,
            color="black",
            linestyle=linestyle,
            marker=marker,
            linewidth=1.6,
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            label=name,
        )

    plt.xlabel("Number of attributes")
    plt.ylabel("Estimated signature size (bytes)")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    plt.legend(fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "signature_size_bytes.png", dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 9. 主流程
# =============================================================================

def parse_attrs(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("属性数量列表不能为空")
    return vals


def run(args: argparse.Namespace) -> None:
    global USE_NUMPY
    USE_NUMPY = bool(args.use_numpy)

    params = Params(
        q=args.q,
        n_poly=args.n,
        zq_dim=args.zq_dim,
        aa_count=args.aa_count,
        revocation_list_size=args.revocation_list_size,
        cmp_vector_len=args.cmp_vector_len,
    )
    schemes = build_schemes()
    out_dir = make_out_dir(args.out)

    timings, timing_rows = measure_all_timings(params, repeat=args.repeat, seed=args.seed)

    param_rows: List[Dict[str, object]] = []
    for s in schemes:
        param_rows.append({
            "scheme": s.name,
            "citation_key": s.citation_key,
            "domain": s.domain,
            "q": params.q,
            "n_poly": params.n_poly if s.domain == "Rq" else "-",
            "zq_dim": params.zq_dim if s.domain == "Zq" else params.equivalent_zq_dim,
            "ell_q": params.ell_q,
            "signature_size": s.size_expr,
            "authority": s.authority,
            "sign_expr": s.sign_expr,
            "verify_expr": s.verify_expr,
        })

    bench_rows: List[Dict[str, object]] = []
    size_rows: List[Dict[str, object]] = []

    for attrs in args.attrs:
        for s in schemes:
            sign_ms = estimate_sign(s, attrs, timings)
            verify_ms = estimate_verify(s, attrs, timings)
            key_total, key_wall = estimate_keygen(s, attrs, timings, params)
            sig_size = estimate_signature_size(s, attrs, params)

            bench_rows.append({
                "scheme": s.name,
                "citation_key": s.citation_key,
                "attrs": attrs,
                "sign_ms": sign_ms,
                "verify_ms": verify_ms,
                "keygen_total_ms": key_total,
                "keygen_wall_ms": key_wall,
                "domain": s.domain,
                "authority": s.authority,
                "sign_expr": s.sign_expr,
                "verify_expr": s.verify_expr,
            })
            size_rows.append({
                "scheme": s.name,
                "attrs": attrs,
                "signature_size_bytes": sig_size,
                "signature_size_expr": s.size_expr,
            })

    safe_write_csv(out_dir / "base_operation_timings.csv", timing_rows)
    safe_write_csv(out_dir / "parameter_table.csv", param_rows)
    safe_write_csv(out_dir / "benchmark_results.csv", bench_rows)
    safe_write_csv(out_dir / "signature_size_results.csv", size_rows)

    safe_write_text(out_dir / "table3_overleaf.tex", table3_overleaf())
    safe_write_text(out_dir / "table3_note_overleaf.tex", table3_note_overleaf())
    safe_write_text(out_dir / "parameter_table_overleaf.tex", params_table_overleaf(params, schemes))

    maybe_plot(out_dir, bench_rows, size_rows)

    print("实验完成。输出目录：", out_dir)
    print("基础操作实测结果：")
    for row in timing_rows:
        print(f"  {row['operation']}: {float(row['mean_ms']):.6f} ms ± {float(row['std_ms']):.6f}")
    print("主要结果：", out_dir / "benchmark_results.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measured-operation benchmark for lattice-based ABS schemes")
    parser.add_argument("--q", type=int, default=8380417, help="模数 q")
    parser.add_argument("--n", type=int, default=256, help="多项式阶 n")
    parser.add_argument("--zq-dim", type=int, default=1024, help="Z_q 等价维度，正式实验建议 1024")
    parser.add_argument("--aa-count", type=int, default=4, help="本文方案属性授权机构数量")
    parser.add_argument("--attrs", type=parse_attrs, default=parse_attrs("4,8,16,32,64,128,256"), help="属性数量列表")
    parser.add_argument("--repeat", type=int, default=5, help="基础操作重复测量次数")
    parser.add_argument("--seed", type=int, default=20260430, help="随机种子")
    parser.add_argument("--revocation-list-size", type=int, default=256, help="撤销列表模拟规模")
    parser.add_argument("--cmp-vector-len", type=int, default=256, help="比较操作模拟向量长度")
    parser.add_argument("--out", type=str, default="results_measured_abs", help="输出目录")
    parser.add_argument("--use-numpy", action="store_true", help="使用 numpy/BLAS 加速 Z_q 矩阵乘法；默认关闭以保持同一实现层级")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
