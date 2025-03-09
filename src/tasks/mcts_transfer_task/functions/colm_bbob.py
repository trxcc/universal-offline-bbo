from __future__ import annotations

"""Experimenter function implementations for BBOB functions.

    Reference:
    [1] Hansen N, Finck S, Ros R, Auger A. Real-parameter black-box optimization 
        benchmarking 2009: Noiseless functions definitions Technical Report, INRIA, 2009.
        https://inria.hal.science/inria-00362633/document

    [2] Wang S, Xue K, Song L, Huang X, Qian C. Monte Carlo tree search based
        space transfer for black-box optimization. In NeurIPS 2024. 
        https://arxiv.org/pdf/2412.07186. 
        Code reference: https://github.com/lamda-bbo/mcts-transfer/blob/main/functions/bbob/bbob.py
    
    [3] Golovin D, Solnik B, Moitra S, Kochanski G, Karro J, Sculley D. Google
        vizier: A service for black-box optimization. In SIGKDD 2017.
        https://dl.acm.org/doi/pdf/10.1145/3097983.3098043.
        Code reference: https://github.com/google/vizier/blob/e4b0842bdea8fb2f36d283c40d2d71fd1087b7b3/vizier/_src/benchmarks/experimenters/synthetic/bbob.py
"""

import hashlib
import math
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats

# from src.tasks.mcts_transfer_task.functions.utils import get_data
# from data.get_data import get_combined_data, get_similar_data


## Utility Functions for BBOB.
def LambdaAlpha(alpha: float, dim: int) -> np.ndarray:
    """The BBOB LambdaAlpha matrix creation function.

    Args:
      alpha: Function parameter.
      dim: Dimension of matrix created.

    Returns:
      Diagonal matrix of dimension dim with values determined by alpha.
    """
    lambda_alpha = np.zeros([dim, dim])
    for i in range(dim):
        exp = (0.5 * (float(i) / (dim - 1))) if dim > 1 else 0.5
        lambda_alpha[i, i] = alpha**exp
    return lambda_alpha


def ArrayMap(vector: np.ndarray, fn: Callable[[float], float]) -> np.ndarray:
    """Create a new array by mapping fn() to each element of the original array.

    Args:
      vector: ndarray to be mapped.
      fn: scalar function for mapping.

    Returns:
      New ndarray be values mapped by fn.
    """
    results = np.zeros(vector.shape)
    for i, v in enumerate(vector.flat):
        results.flat[i] = fn(v)
    return results


def Tosz(element: float) -> float:
    """The BBOB T_osz function.

    Args:
      element: float input.

    Returns:
      Tosz(input).
    """
    x_carat = 0.0 if element == 0 else math.log(abs(element))
    c1 = 10.0 if element > 0 else 5.5
    c2 = 7.9 if element > 0 else 3.1
    return np.sign(element) * math.exp(
        x_carat + 0.049 * (math.sin(c1 * x_carat) + math.sin(c2 * x_carat))
    )


def Tasy(vector: np.ndarray, beta: float) -> np.ndarray:
    """The BBOB Tasy function.

    Args:
      vector: ndarray
      beta: Function parameter

    Returns:
      ndarray with values determined by beta.
    """
    dim = len(vector)
    result = np.zeros([dim, 1])
    for i, val in enumerate(vector.flat):
        if val > 0:
            t = i / (dim - 1.0) if dim > 1 else 1
            exp = 1 + beta * t * (val**0.5)
        else:
            exp = 1
        result[i] = val**exp
    return result


def SIndex(dim: int, to_sz) -> float:
    """Calculate the BBOB s_i.

    Assumes i is 0-index based.

    Args:
      dim: dimension
      to_sz: values

    Returns:
      float representing SIndex(i, d, to_sz).
    """
    s = np.zeros(
        [
            dim,
        ]
    )
    for i in range(dim):
        if dim > 1:
            s[i] = 10 ** (0.5 * (i / (dim - 1.0)))
        else:
            s[i] = 10**0.5
        if i % 2 == 0 and to_sz[i] > 0:
            s[i] *= 10
    return s


def Fpen(vector: np.ndarray) -> float:
    """The BBOB Fpen function.

    Args:
      vector: ndarray.

    Returns:
      float representing Fpen(vector).
    """
    return sum([max(0.0, (abs(x) - 5.0)) ** 2 for x in vector.flat])


def _IntSeeds(any_seeds: Sequence[Any], *, byte_length: int = 4) -> list[int]:
    """Array of integers that can be used as random state seed."""
    int_seeds = []
    for s in any_seeds:
        # Encode into 4 byte_length worth of a hexadecimal string.
        hashed = hashlib.shake_128(str(s).encode("utf-8")).hexdigest(byte_length)
        int_seeds.append(int(hashed, 16))
    return int_seeds


def _ToFloat(a: int, b: np.ndarray) -> np.ndarray:
    """Convert a%b where b is an int into a float on [-0.5, 0.5]."""
    return (np.int64(a) % b) / np.float64(b) - 0.5


def _R(dim: int, seed: int, *moreseeds: Any) -> np.ndarray:
    """
    Returns an orthonormal rotation matrix.

    Args:
      dim: size of the resulting matrix.
      seed: int seed. If set to 0, this function returns an identity matrix
        regardless of *moreseeds.
      *moreseeds: Additional parameters to include in the hash. Arguments are
        converted to strings first.

    Returns:
      Array of shape (dim, dim), representing a rotation matrix.
    """
    if seed == 0:
        return np.identity(dim)
    rng = np.random.default_rng(_IntSeeds(((seed, dim) + moreseeds)))
    return stats.special_ortho_group.rvs(dim, random_state=rng)


## BBOB Functions.
    
class Sphere:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        arr = arr - self.c
        return float(np.sum(arr * arr))

class Rastrigin:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        # print(arr.shape)
        arr.shape = (dim, 1)
        # print(arr.shape)
        # print(self.c.shape)
        arr = arr - self.c
        # print(arr.shape)
        # assert 0
        z = np.matmul(_R(dim, seed, b"R"), arr)
        z = Tasy(ArrayMap(z, Tosz), 0.2)
        z = np.matmul(_R(dim, seed, b"Q"), z)
        z = np.matmul(LambdaAlpha(10.0, dim), z)
        z = np.matmul(_R(dim, seed, b"R"), z)
        return float(10 * (dim - np.sum(np.cos(2 * math.pi * z))) +
               np.sum(z * z, axis=0))

class BuecheRastrigin:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        del seed
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        t = ArrayMap(arr, Tosz)
        l = SIndex(dim, arr) * t.flat

        term1 = 10 * (dim - np.sum(np.cos(2 * math.pi * l), axis=0))
        term2 = np.sum(l * l, axis=0)
        term3 = 100 * Fpen(arr)
        return float(term1 + term2 + term3)
    
class LinearSlope:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        r = _R(dim, seed, b"R")
        z = np.matmul(r, arr)
        result = 0.0
        for i in range(dim):
            s = 10**(i / float(dim - 1) if dim > 1 else 1)
            z_opt = 5 * np.sum(np.abs(r[i, :]))
            result += float(s * (z_opt - z[i]))
        return result

class AttractiveSector:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        x_opt = np.array([1 if i % 2 == 0 else -1 for i in range(dim)])
        x_opt.shape = (dim, 1)
        z_vec = np.matmul(_R(dim, seed, b"R"), arr - x_opt)
        z_vec = np.matmul(LambdaAlpha(10.0, dim), z_vec)
        z_vec = np.matmul(_R(dim, seed, b"Q"), z_vec)

        result = 0.0
        for i in range(dim):
            z = z_vec[i, 0]
            s = 100 if z * x_opt[i] > 0 else 1
            result += (s * z)**2

        return math.pow(Tosz(result), 0.9)
    
class StepEllipsoidal:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        z_hat = np.matmul(_R(dim, seed, b"R"), arr)
        z_hat = np.matmul(LambdaAlpha(10.0, dim), z_hat)
        z_tilde = np.array([
            math.floor(0.5 + z) if (z > 0.5) else (math.floor(0.5 + 10 * z) / 10)
            for z in z_hat.flat
        ])
        z_tilde = np.matmul(_R(dim, seed, b"Q"), z_tilde)
        s = 0.0
        for i, val in enumerate(z_tilde):
            exponent = 2.0 * float(i) / (dim - 1.0) if dim > 1.0 else 2.0
            s += 10.0**exponent * val**2
        value = max(abs(z_hat[0, 0]) / 1000, s)
        return 0.1 * value + Fpen(arr)
    
class RosenbrockRotated:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        # self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr = arr - self.c
        r_x = np.matmul(_R(dim, seed, b"R"), arr)
        z = max(1.0, (dim**0.5) / 8.0) * r_x + 0.5 * np.ones((dim,))
        return float(
            sum([
                100.0 * (z[i]**2 - z[i + 1])**2 + (z[i] - 1)**2
                for i in range(dim - 1)
            ]))

class Ellipsoidal:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        del seed
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        z_vec = ArrayMap(arr, Tosz)
        s = 0.0
        for i in range(dim):
            exp = 6.0 * i / (dim - 1) if dim > 1 else 6.0
            s += float(10**exp * z_vec[i] * z_vec[i])
        return s

class Discus:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        r_x = np.matmul(_R(dim, seed, b"R"), arr)
        z_vec = ArrayMap(r_x, Tosz)
        return float(10**6 * z_vec[0] * z_vec[0]) + sum(
            [z * z for z in z_vec[1:].flat])

class BentCigar:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        z_vec = np.matmul(_R(dim, seed, b"R"), arr)
        z_vec = Tasy(z_vec, 0.5)
        z_vec = np.matmul(_R(dim, seed, b"R"), z_vec)
        return float(z_vec[0]**2) + 10**6 * np.sum(z_vec[1:]**2)

class SharpRidge:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        z_vec = np.matmul(_R(dim, seed, b"R"), arr)
        z_vec = np.matmul(LambdaAlpha(10, dim), z_vec)
        z_vec = np.matmul(_R(dim, seed, b"Q"), z_vec)
        return z_vec[0, 0]**2 + 100 * np.sum(z_vec[1:]**2)**0.5
    
class DifferentPowers:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        # self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr = arr - self.c
        z = np.matmul(_R(dim, seed, b"R"), arr)
        s = 0.0
        for i in range(dim):
            exp = 2 + 4 * i / (dim - 1) if dim > 1 else 6
            s += abs(z[i])**exp
        return s**0.5

class Weierstrass:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        k_order = 12
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        z = np.matmul(_R(dim, seed, b"R"), arr)
        z = ArrayMap(z, Tosz)
        z = np.matmul(_R(dim, seed, b"Q"), z)
        z = np.matmul(LambdaAlpha(1.0 / 100.0, dim), z)
        f0 = sum([0.5**k * math.cos(math.pi * 3**k) for k in range(k_order)])

        s = 0.0
        for i in range(dim):
            for k in range(k_order):
                s += 0.5**k * math.cos(2 * math.pi * (3**k) * (z[i] + 0.5))

        return float(10 * (s / dim - f0)**3) + 10 * Fpen(arr) / dim
    
class SchaffersF7:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        if dim == 1:
            return 0.0
        arr = arr - self.c
        z = np.matmul(_R(dim, seed, b"R"), arr)
        z = Tasy(z, 0.5)
        z = np.matmul(_R(dim, seed, b"Q"), z)
        z = np.matmul(LambdaAlpha(10.0, dim), z)

        s_arr = np.zeros(dim - 1)
        for i in range(dim - 1):
            s_arr[i] = float((z[i]**2 + z[i + 1]**2)**0.5)
        s = 0.0
        for i in range(dim - 1):
            s += s_arr[i]**0.5 + (s_arr[i]**0.5) * math.sin(50 * s_arr[i]**0.2)**2

        return (s / (dim - 1.0))**2 + 10 * Fpen(arr)

class SchaffersF7IllConditioned:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        if dim == 1:
            return 0.0
        z = np.matmul(_R(dim, seed, b"R"), arr)
        z = Tasy(z, 0.5)
        z = np.matmul(_R(dim, seed, b"Q"), z)
        z = np.matmul(LambdaAlpha(1000.0, dim), z)

        s_arr = np.zeros(dim - 1)
        for i in range(dim - 1):
            s_arr[i] = float((z[i]**2 + z[i + 1]**2)**0.5)
        s = 0.0
        for i in range(dim - 1):
            s += s_arr[i]**0.5 + (s_arr[i]**0.5) * math.sin(50 * s_arr[i]**0.2)**2

        return (s / (dim - 1.0))**2 + 10 * Fpen(arr)

class GriewankRosenbrock:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        # self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr = arr - self.c
        r_x = np.matmul(_R(dim, seed, b"R"), arr)
        # Slightly off BBOB documentation in order to center optima at origin.
        # Should be: max(1.0, (dim**0.5) / 8.0) * r_x + 0.5 * np.ones((dim,)).
        z_arr = max(1.0, (dim**0.5) / 8.0) * r_x + np.ones((dim,))
        s_arr = np.zeros(dim)
        for i in range(dim - 1):
            s_arr[i] = 100.0 * (z_arr[i]**2 - z_arr[i + 1])**2 + (z_arr[i] - 1)**2

        total = 0.0
        for i in range(dim - 1):
            total += (s_arr[i] / 4000.0 - math.cos(s_arr[i]))

        return (10.0 * total) / (dim - 1) + 10

class Schwefel:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        # self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        del seed
        dim = len(arr)
        arr = arr - self.c
        bernoulli_arr = np.array([pow(-1, i + 1) for i in range(dim)])
        x_opt = 4.2096874633 / 2.0 * bernoulli_arr
        x_hat = 2.0 * (bernoulli_arr * arr)  # Element-wise multiplication

        z_hat = np.zeros([dim, 1])
        z_hat[0, 0] = x_hat[0]
        for i in range(1, dim):
            z_hat[i, 0] = x_hat[i] + 0.25 * (x_hat[i - 1] - 2 * abs(x_opt[i - 1]))

        x_opt.shape = (dim, 1)
        z_vec = 100 * (
            np.matmul(LambdaAlpha(10, dim), z_hat - 2 * abs(x_opt)) + 2 * abs(x_opt))

        total = sum([z * math.sin(abs(z)**0.5) for z in z_vec.flat])

        return -(total / (100.0 * dim)) + 4.189828872724339 + 100 * Fpen(z_vec / 100)

class Katsuura:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        r_x = np.matmul(_R(dim, seed, b"R"), arr)
        z_vec = np.matmul(LambdaAlpha(100.0, dim), r_x)
        z_vec = np.matmul(_R(dim, seed, b"Q"), z_vec)

        prod = 1.0
        for i in range(dim):
            s = 0.0
            for j in range(1, 33):
                s += abs(2**j * z_vec[i, 0] - round(2**j * z_vec[i, 0])) / 2**j
            prod *= (1 + (i + 1) * s)**(10.0 / dim**1.2)

        return (10.0 / dim**2) * prod - 10.0 / dim**2 + Fpen(arr)

class Lunacek:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        mu0 = 2.5
        s = 1.0 - 1.0 / (2.0 * (dim + 20.0)**0.5 - 8.2)
        mu1 = -((mu0**2 - 1) / s)**0.5

        x_opt = np.array([mu0 / 2] * dim)
        x_hat = np.array([2 * arr[i, 0] * np.sign(x_opt[i]) for i in range(dim)])
        x_vec = x_hat - mu0
        x_vec.shape = (dim, 1)
        x_vec = np.matmul(_R(dim, seed, b"R"), x_vec)
        z_vec = np.matmul(LambdaAlpha(100, dim), x_vec)
        z_vec = np.matmul(_R(dim, seed, b"Q"), z_vec)

        s1 = sum([(val - mu0)**2 for val in x_hat])
        s2 = sum([(val - mu1)**2 for val in x_hat])
        s3 = sum([math.cos(2 * math.pi * z) for z in z_vec.flat])
        return min(s1, dim + s * s2) + 10.0 * (dim - s3) + 10**4 * Fpen(arr)

class Gallagher101Me:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c

        num_optima = 101
        optima_list = [np.zeros([dim, 1])]
        for i in range(num_optima - 1):
            vec = np.zeros([dim, 1])
            for j in range(dim):
                alpha = (i * dim + j + 1.0) / (dim * num_optima + 2.0)
                assert alpha > 0
                assert alpha < 1
                vec[j, 0] = -5 + 10 * alpha
            optima_list.append(vec)

        c_list = [LambdaAlpha(1000, dim)]
        for i in range(num_optima - 1):
            alpha = 1000.0**(2.0 * (i) / (num_optima - 2))
            c_mat = LambdaAlpha(alpha, dim) / (alpha**0.25)
            c_list.append(c_mat)

        rotation = _R(dim, seed, b"R")
        max_value = -1.0
        for i in range(num_optima):
            w = 10 if i == 0 else (1.1 + 8.0 * (i - 1.0) / (num_optima - 2.0))
            diff = np.matmul(rotation, arr - optima_list[i])
            e = np.matmul(diff.transpose(), np.matmul(c_list[i], diff))
            max_value = max(max_value, w * math.exp(-float(e) / (2.0 * dim)))

        return Tosz(10.0 - max_value)**2 + Fpen(arr)

class Gallagher21Me:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c

        num_optima = 21
        optima_list = [np.zeros([dim, 1])]
        for i in range(num_optima - 1):
            vec = np.zeros([dim, 1])
            for j in range(dim):
                alpha = (i * dim + j + 1.0) / (dim * num_optima + 2.0)
                assert alpha > 0
                assert alpha < 1
                vec[j, 0] = -5 + 10 * alpha
            optima_list.append(vec)

        c_list = [LambdaAlpha(1000, dim)]
        for i in range(num_optima - 1):
            alpha = 1000.0**(2.0 * (i) / (num_optima - 2))
            c_mat = LambdaAlpha(alpha, dim) / (alpha**0.25)
            c_list.append(c_mat)

        rotation = _R(dim, seed, b"R")
        max_value = -1.0
        for i in range(num_optima):
            w = 10 if i == 0 else (1.1 + 8.0 * (i - 1.0) / (num_optima - 2.0))
            diff = np.matmul(rotation, arr - optima_list[i])
            e = np.matmul(diff.transpose(), np.matmul(c_list[i], diff))
            max_value = max(max_value, w * math.exp(-float(e) / (2.0 * dim)))

        return Tosz(10.0 - max_value)**2 + Fpen(arr)

## Additional BBOB-like functions to test exploration.

class NegativeSphere:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        z = np.matmul(_R(dim, seed, b"R"), arr)
        return float(100 + np.sum(z * z) - 2 * (z[0]**2))

class NegativeMinDifference:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        dim = len(arr)
        arr.shape = (dim, 1)
        arr = arr - self.c
        z = np.matmul(_R(dim, seed, b"R"), arr)
        min_difference = 10000
        for i in range(len(z) - 1):
            min_difference = min(min_difference, z[i + 1] - z[i])
        return 10.0 - float(min_difference) + 1e-8 * float(sum(arr))

class FonsecaFleming:
    def __init__(self, dim):
        self.c = np.random.uniform(-3, 3, dim)
        # self.c.shape = (dim, 1)
        
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        del seed
        arr = arr - self.c
        return 1.0 - float(np.exp(-np.sum(arr * arr)))


if __name__ == "__main__":
    problem = SharpRidge()
    print(problem(np.ones(10)))
