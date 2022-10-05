from __future__ import annotations

import typing

import numpy as np
from numpy.typing import NDArray
from scipy import sparse  # type: ignore
from scipy.special import factorial  # type: ignore


class UniformPeriodicGrid:
    def __init__(self, N: int, length: float):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:
    def __init__(self, values: NDArray[np.float64], length: float):
        self.values = values
        self.length = length
        self.N = len(values)


class DifferenceUniformGrid:
    def __init__(
        self,
        derivative_order: int,
        convergence_order: int,
        grid: UniformPeriodicGrid,
        stencil_type: str = "centered",
    ):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.matrix = DifferenceNonUniformGrid(
            derivative_order,
            convergence_order,
            NonUniformPeriodicGrid(grid.values, grid.length),
            stencil_type,
        ).matrix

    def __matmul__(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        return typing.cast(NDArray[np.float64], self.matrix @ other)


class DifferenceNonUniformGrid:
    def __init__(
        self,
        derivative_order: int,
        convergence_order: int,
        grid: NonUniformPeriodicGrid,
        stencil_type: str = "centered",
    ):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        num_points = derivative_order + convergence_order + 1
        matrix = sparse.dok_matrix((grid.N, grid.N))
        for i in range(*grid.values.shape):  # 以这个点为中心
            xi = grid.values[i]
            rng = range(i - num_points // 2, i + num_points // 2 + 1)
            for j in rng:  # 遍历所有的插值点
                poly = np.zeros((derivative_order + 1,))
                poly[0] = 1
                xj = (j // grid.N) * grid.length + grid.values[j % grid.N] - xi
                for k in rng:  # 遍历所有的Lagrange单项式
                    if k != j:
                        xk = (k // grid.N) * grid.length + grid.values[k % grid.N] - xi
                        poly[1:] = (xk * poly[1:] - poly[:-1]) / (xk - xj)
                        poly[0] *= xk / (xk - xj)
                matrix[i, j % grid.N] += poly[-1]
        matrix = matrix.tocsr()
        matrix *= factorial(derivative_order)
        self.matrix = matrix

    def __matmul__(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        return typing.cast(NDArray[np.float64], self.matrix @ other)
