import math

import numpy as np
import pytest
from numpy.typing import NDArray

import finite
import timesteppers

resolution_list = [100, 200, 400]

error_RK_2_2 = {100: 0.5, 200: 0.15, 400: 0.05}


@pytest.mark.parametrize("resolution", resolution_list)
def test_RK_2_2(resolution: int) -> None:
    grid = finite.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values
    IC = np.exp(-((x - np.pi) ** 2) * 8)

    target = np.exp(-((x - np.pi - 2 * np.pi * 0.2) ** 2) * 8)

    d = finite.CenteredFiniteDifference(grid)

    def f(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return d @ u

    stages = 2
    a = np.array([[0, 0], [1 / 2, 0]])
    b = np.array([0, 1])

    ts = timesteppers.Multistage(IC, f, stages, a, b)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(alpha * grid.dx, 2 * np.pi * num_periods)

    error = np.abs(ts.u - target).max()
    error_est = error_RK_2_2[resolution]

    assert error < error_est


error_RK_2_4 = {100: 0.15, 200: 0.05, 400: 0.01}


@pytest.mark.parametrize("resolution", resolution_list)
def test_RK_2_4(resolution: int) -> None:
    grid = finite.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values
    IC = np.exp(-((x - np.pi) ** 2) * 8)

    target = np.exp(-((x - np.pi + 2 * np.pi * 0.2) ** 2) * 8)

    d = finite.CenteredFiniteDifference4(grid)

    def f(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return d @ u

    stages = 2
    a = np.array([[0, 0], [1 / 2, 0]])
    b = np.array([0, 1])

    ts = timesteppers.Multistage(IC, f, stages, a, b)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(alpha * grid.dx, 2 * np.pi * num_periods)

    error = np.abs(ts.u - target).max()
    error_est = error_RK_2_4[resolution]

    assert error < error_est


error_RK_3_2 = {100: 0.5, 200: 0.2, 400: 0.05}


@pytest.mark.parametrize("resolution", resolution_list)
def test_RK_3_2(resolution: int) -> None:
    grid = finite.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values
    IC = np.exp(-((x - np.pi) ** 2) * 8)

    target = np.exp(-((x - np.pi - 2 * np.pi * 0.2) ** 2) * 8)

    d = finite.CenteredFiniteDifference(grid)

    def f(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return d @ u

    stages = 3
    a = np.array([[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]])
    b: NDArray[np.float64] = np.array([1, 4, 1]) / 6

    ts = timesteppers.Multistage(IC, f, stages, a, b)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(alpha * grid.dx, 2 * np.pi * num_periods)

    error = np.abs(ts.u - target).max()
    error_est = error_RK_3_2[resolution]

    assert error < error_est


error_RK_3_4 = {100: 0.04, 200: 0.005, 400: 3e-4}


@pytest.mark.parametrize("resolution", resolution_list)
def test_RK_3_4(resolution: int) -> None:
    grid = finite.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values
    IC = np.exp(-((x - np.pi) ** 2) * 8)

    target = np.exp(-((x - np.pi + 2 * np.pi * 0.2) ** 2) * 8)

    d = finite.CenteredFiniteDifference4(grid)

    def f(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return d @ u

    stages = 3
    a = np.array([[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]])
    b: NDArray[np.float64] = np.array([1, 4, 1]) / 6

    ts = timesteppers.Multistage(IC, f, stages, a, b)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(alpha * grid.dx, 2 * np.pi * num_periods)

    error = np.abs(ts.u - target).max()
    error_est = error_RK_3_4[resolution]

    assert error < error_est


error_RK_4_2 = {100: 0.5, 200: 0.2, 400: 0.05}


@pytest.mark.parametrize("resolution", resolution_list)
def test_RK_4_2(resolution: int) -> None:
    grid = finite.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values
    IC = np.exp(-((x - np.pi) ** 2) * 8)

    target = np.exp(-((x - np.pi - 2 * np.pi * 0.2) ** 2) * 8)

    d = finite.CenteredFiniteDifference(grid)

    def f(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return d @ u

    stages = 4
    a = np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
    b: NDArray[np.float64] = np.array([1, 2, 2, 1]) / 6

    ts = timesteppers.Multistage(IC, f, stages, a, b)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(alpha * grid.dx, 2 * np.pi * num_periods)

    error = np.abs(ts.u - target).max()
    error_est = error_RK_4_2[resolution]

    assert error < error_est


error_RK_4_4 = {100: 0.04, 200: 0.003, 400: 2e-4}


@pytest.mark.parametrize("resolution", resolution_list)
def test_RK_4_4(resolution: int) -> None:
    grid = finite.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values
    IC = np.exp(-((x - np.pi) ** 2) * 8)

    target = np.exp(-((x - np.pi + 2 * np.pi * 0.2) ** 2) * 8)

    d = finite.CenteredFiniteDifference4(grid)

    def f(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return d @ u

    stages = 4
    a = np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
    b: NDArray[np.float64] = np.array([1, 2, 2, 1]) / 6

    ts = timesteppers.Multistage(IC, f, stages, a, b)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(alpha * grid.dx, 2 * np.pi * num_periods)

    error = np.abs(ts.u - target).max()
    error_est = error_RK_4_4[resolution]

    assert error < error_est


@pytest.mark.parametrize("resolution", resolution_list)
def test_Adams_2_2(resolution: int) -> None:
    grid = finite.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values
    IC = np.exp(-((x - np.pi) ** 2) * 8)

    target = np.exp(-((x - np.pi - 2 * np.pi * 0.2) ** 2) * 8)

    d = finite.CenteredFiniteDifference(grid)

    def f(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return d @ u

    stages = 3

    ts = timesteppers.AdamsBashforth(IC, f, stages, None)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(alpha * grid.dx, 2 * np.pi * num_periods)

    error = np.abs(ts.u - target).max()
    error_est = error_RK_2_2[resolution]

    assert error < error_est * 1.1


@pytest.mark.parametrize("resolution", resolution_list)
def test_Adams_2_4(resolution: int) -> None:
    grid = finite.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values
    IC = np.exp(-((x - np.pi) ** 2) * 8)

    target = np.exp(-((x - np.pi + 2 * np.pi * 0.2) ** 2) * 8)

    d = finite.CenteredFiniteDifference4(grid)

    def f(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return d @ u

    stages = 3

    ts = timesteppers.AdamsBashforth(IC, f, stages, None)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(alpha * grid.dx, 2 * np.pi * num_periods)

    error = np.abs(ts.u - target).max()
    error_est = error_RK_2_4[resolution]

    assert error < error_est


def test_Adams_2() -> None:
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([x[1], -x[0]])

    IC = np.array([1.0, 0.0])
    stages = 2
    ts = timesteppers.AdamsBashforth(IC, f, stages, None)
    ts.evolve(0.1, 10)
    target = math.cos(10)
    error = abs(ts.u[0] - target)
    assert error < 1.8e-2


def test_Adams_3() -> None:
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([x[1], -x[0]])

    IC = np.array([1.0, 0.0])
    stages = 3
    ts = timesteppers.AdamsBashforth(IC, f, stages, None)
    ts.evolve(0.1, 10)
    target = math.cos(10)
    error = abs(ts.u[0] - target)
    assert error < 1e-2


def test_Adams_4() -> None:
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([x[1], -x[0]])

    IC = np.array([1.0, 0.0])
    stages = 4
    ts = timesteppers.AdamsBashforth(IC, f, stages, None)
    ts.evolve(0.1, 10)
    target = math.cos(10)
    error = abs(ts.u[0] - target)
    assert error < 1e-2


def test_Adams_5() -> None:
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([x[1], -x[0]])

    IC = np.array([1.0, 0.0])
    stages = 5
    ts = timesteppers.AdamsBashforth(IC, f, stages, None)
    ts.evolve(0.1, 10)
    target = math.cos(10)
    error = abs(ts.u[0] - target)
    assert error < 1e-2


def test_Adams_6() -> None:
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([x[1], -x[0]])

    IC = np.array([1.0, 0.0])
    stages = 6
    ts = timesteppers.AdamsBashforth(IC, f, stages, None)
    ts.evolve(0.1, 10)
    target = math.cos(10)
    error = abs(ts.u[0] - target)
    assert error < 1e-2
