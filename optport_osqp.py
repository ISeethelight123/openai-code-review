"""
Portfolio optimization utilities.

Author : Kevin (refactored)
Created: 2019-09-09
Refactor: OSQP 替代 cvxpy；Ray 按批并行（每批只 put 本批切片，避免大表进 worker 崩进程）。

说明
----
实践结论：把完整多日大表一次 put 后在 worker 内再切日 + OSQP，
即使 batch_size=1 也会把 Ray worker/会话打挂。
能稳定跑通的方式与早期 Ray 版一致：驱动侧按日切好，
每个任务只接收「本批」小数据。

Dependencies
------------
osqp==0.6.7.post3 / scipy / numpy / pandas
ray（项目内已初始化：from ray.util.client import ray）
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp
from ray.util.client import ray

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
_OSQP_INF = 1e20


def _solve_one_day(
    codealpha: pd.DataFrame,
    dotmat: Optional[pd.DataFrame],
    dotlowerbond: Optional[pd.DataFrame],
    dotupperbond: Optional[pd.DataFrame],
    mulmat: Optional[pd.DataFrame],
    mullowerbond: Optional[pd.DataFrame],
    mulupperbond: Optional[pd.DataFrame],
    minwt: Optional[float],
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """单日 OSQP 求解 + 旧版对齐的权重后处理（不含 fallback / 删行）。"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    codealpha_idx = codealpha.set_index("code")
    N = int(codealpha_idx.shape[0])
    if N <= 0:
        return None, None

    alpha_vals = np.nan_to_num(
        codealpha_idx["alpha"].to_numpy(dtype=np.float64), nan=0.0
    )
    q = np.ascontiguousarray(-alpha_vals, dtype=np.float64)

    # 极小对角正则 + 上三角 CSC，避免全零 P 在部分环境不稳
    P = sp.triu(1e-8 * sp.eye(N, format="csc", dtype=np.float64), format="csc")

    A_blocks: list[sp.spmatrix] = []
    l_blocks: list[np.ndarray] = []
    u_blocks: list[np.ndarray] = []

    A_blocks.append(sp.eye(N, format="csc", dtype=np.float64))
    l_blocks.append(np.zeros(N, dtype=np.float64))
    u_blocks.append(np.full(N, _OSQP_INF, dtype=np.float64))

    A_blocks.append(sp.csc_matrix(np.ones((1, N), dtype=np.float64)))
    l_blocks.append(np.array([1.0], dtype=np.float64))
    u_blocks.append(np.array([1.0], dtype=np.float64))

    if dotmat is not None and dotlowerbond is not None and dotupperbond is not None:
        dm = np.nan_to_num(dotmat.to_numpy(dtype=np.float64), nan=0.0)
        A_blocks.append(sp.csc_matrix(dm.T, dtype=np.float64))
        l_blocks.append(
            np.nan_to_num(
                dotlowerbond.to_numpy(dtype=np.float64).flatten(), nan=-_OSQP_INF
            )
        )
        u_blocks.append(
            np.nan_to_num(
                dotupperbond.to_numpy(dtype=np.float64).flatten(), nan=_OSQP_INF
            )
        )

    if mulmat is not None and mullowerbond is not None and mulupperbond is not None:
        mv = np.nan_to_num(mulmat.to_numpy(dtype=np.float64).flatten(), nan=1.0)
        A_blocks.append(sp.diags(mv, format="csc", dtype=np.float64))
        l_blocks.append(
            np.nan_to_num(
                mullowerbond.to_numpy(dtype=np.float64).flatten(), nan=-_OSQP_INF
            )
        )
        u_blocks.append(
            np.nan_to_num(
                mulupperbond.to_numpy(dtype=np.float64).flatten(), nan=_OSQP_INF
            )
        )

    A = sp.vstack(A_blocks, format="csc").astype(np.float64)
    l = np.ascontiguousarray(np.concatenate(l_blocks), dtype=np.float64)
    u = np.ascontiguousarray(np.concatenate(u_blocks), dtype=np.float64)

    l = np.where(np.isneginf(l), -_OSQP_INF, l)
    l = np.where(np.isposinf(l), _OSQP_INF, l)
    u = np.where(np.isneginf(u), -_OSQP_INF, u)
    u = np.where(np.isposinf(u), _OSQP_INF, u)
    bad = l > u
    if np.any(bad):
        mid = 0.5 * (l[bad] + u[bad])
        l[bad] = mid
        u[bad] = mid

    solver = osqp.OSQP()
    try:
        solver.setup(
            P, q, A, l, u,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=10000,
            polish=False,
            adaptive_rho=False,
            check_termination=25,
        )
        result = solver.solve()
    except Exception as e:
        logger.error(f"OSQP 求解异常: {e}")
        return None, None

    status: str = result.info.status
    if result.x is None or status not in ("solved", "solved_inaccurate"):
        logger.warning(f"OSQP 未求解，状态: {status}")
        return None, status

    tmp_weight = (
        pd.Series(result.x.flatten(), index=codealpha_idx.index).to_frame(name="wt")
    )
    tmp_weight[tmp_weight <= 0] = np.nan

    if (
        (tmp_weight["wt"] <= 1e-5).all()
        or np.isinf(tmp_weight.values).any()
        or tmp_weight["wt"].isna().all()
        or (tmp_weight["wt"].max() > 1e20)
    ):
        logger.warning("OSQP 解异常（全零/inf/nan/过大），判定失败")
        return None, status

    if minwt is not None:
        tmp_weight[np.abs(tmp_weight) <= minwt] = np.nan
    else:
        tmp_weight = tmp_weight.sort_values(by=["wt"], ascending=[False])
        tmp_weight = tmp_weight[["wt"]].copy()

    tmp_weight = tmp_weight / np.sum(tmp_weight)
    tmp_weight = tmp_weight.replace(np.nan, 0)
    tmp_weight = tmp_weight.reset_index()
    tmp_weight.columns = ["code", "wt"]
    return tmp_weight, status


def _slice_one_day(
    optdt: Any,
    datecodealpha: pd.DataFrame,
    dotmat: Optional[pd.DataFrame],
    dotlowerbond: Optional[pd.DataFrame],
    dotupperbond: Optional[pd.DataFrame],
    mulmat: Optional[pd.DataFrame],
    mullowerbond: Optional[pd.DataFrame],
    mulupperbond: Optional[pd.DataFrame],
) -> dict[str, Any]:
    """驱动侧按日切片（与旧 getoptport_ 一致）。"""
    codealpha = (
        datecodealpha.loc[datecodealpha["date"] == optdt, ["code", "alpha"]]
        .drop_duplicates("code")
        .reset_index(drop=True)
        .copy()
    )
    codes = sorted(codealpha["code"].tolist())

    dotmatdt = dotlowerbonddt = dotupperbonddt = None
    if dotmat is not None:
        dotmatdt = dotmat.loc[[optdt]].reset_index(level="date", drop=True)
        dotmatdt = dotmatdt[~dotmatdt.index.duplicated()].reindex(codes)
        dotlowerbonddt = dotlowerbond.loc[[optdt]]
        dotupperbonddt = dotupperbond.loc[[optdt]]

    mulmatdt = mullowerbonddt = mulupperbonddt = None
    if mulmat is not None:
        mulmatdt = mulmat.loc[[optdt]].reset_index(level="date", drop=True)
        mulmatdt = mulmatdt[~mulmatdt.index.duplicated()].reindex(codes)
        mullowerbonddt = (
            mullowerbond.loc[[optdt]]
            .reset_index(level="date", drop=True)
        )
        mullowerbonddt = mullowerbonddt[~mullowerbonddt.index.duplicated()].reindex(codes)
        mulupperbonddt = (
            mulupperbond.loc[[optdt]]
            .reset_index(level="date", drop=True)
        )
        mulupperbonddt = mulupperbonddt[~mulupperbonddt.index.duplicated()].reindex(codes)

    return {
        "optdt": optdt,
        "codealpha": codealpha,
        "dotmat": dotmatdt,
        "dotlowerbond": dotlowerbonddt,
        "dotupperbond": dotupperbonddt,
        "mulmat": mulmatdt,
        "mullowerbond": mullowerbonddt,
        "mulupperbond": mulupperbonddt,
    }


@ray.remote
def _getdtoptport_(
    batch_days: list[dict[str, Any]],
    minwt: Optional[float],
    mujieat: float,
) -> pd.DataFrame:
    """
    Ray remote：处理一批「已切好」的交易日数据。

    输入仅为本批小对象，不接收完整多日大表。
    内含求解 + 失败 fallback + 噪声清理。
    返回 columns=[date, code, wt]。
    """
    portwtlist: list[pd.DataFrame] = []

    for day in batch_days:
        optdt = day["optdt"]
        codealpha = day["codealpha"]

        portwttemp, _status = _solve_one_day(
            codealpha=codealpha,
            dotmat=day["dotmat"],
            dotlowerbond=day["dotlowerbond"],
            dotupperbond=day["dotupperbond"],
            mulmat=day["mulmat"],
            mullowerbond=day["mullowerbond"],
            mulupperbond=day["mulupperbond"],
            minwt=minwt,
        )

        if portwttemp is None:
            codealpha = codealpha.sort_values(
                "alpha", ascending=False
            ).reset_index(drop=True)
            codealpha["alpharank"] = codealpha["alpha"].rank(
                ascending=False, method="min"
            )
            portwttemp = codealpha.loc[
                codealpha["alpharank"] <= 1 / mujieat, ["code", "alpha"]
            ].copy()
            if portwttemp.shape[0] > 0:
                portwttemp["wt"] = 1 / portwttemp.shape[0]
        else:
            if minwt is not None and minwt != 0:
                portwttemp = (
                    portwttemp[portwttemp["wt"] >= minwt]
                    .reset_index(drop=True)
                    .copy()
                )
            elif minwt == 0 or minwt is None:
                portwttemp = (
                    portwttemp[portwttemp["wt"] > 0.00001]
                    .reset_index(drop=True)
                    .copy()
                )

        portwttemp["date"] = optdt
        portwtlist.append(portwttemp[["date", "code", "wt"]])

    if not portwtlist:
        return pd.DataFrame(columns=["date", "code", "wt"])
    return pd.concat(portwtlist, axis=0, ignore_index=True)


class OptPortClass:
    """组合优化类（OSQP + Ray 按批并行）。"""

    def getoptport_(
        self,
        datecodealpha: pd.DataFrame,
        dotmat: Optional[pd.DataFrame] = None,
        dotlowerbond: Optional[pd.DataFrame] = None,
        dotupperbond: Optional[pd.DataFrame] = None,
        mulmat: Optional[pd.DataFrame] = None,
        mullowerbond: Optional[pd.DataFrame] = None,
        mulupperbond: Optional[pd.DataFrame] = None,
        minwt: Optional[float] = None,
        mujieat: float = 0.05,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[float]]:
        """
        驱动侧按日切片 → 每批只 put 本批数据 → Ray 并行求解 → 汇总。
        """
        if batch_size <= 0:
            raise ValueError("batch_size 必须 > 0")

        datecodealpha = datecodealpha.copy()
        if isinstance(datecodealpha.index, pd.MultiIndex):
            datecodealpha = datecodealpha.reset_index()
        datecodealpha.columns = ["date", "code", "alpha"]

        optperiods = sorted(datecodealpha["date"].unique())

        # 驱动侧切好全部交易日（小对象列表）；不把整表塞进单个 remote
        all_days = [
            _slice_one_day(
                optdt,
                datecodealpha,
                dotmat,
                dotlowerbond,
                dotupperbond,
                mulmat,
                mullowerbond,
                mulupperbond,
            )
            for optdt in optperiods
        ]

        batches = [
            all_days[i : i + batch_size]
            for i in range(0, len(all_days), batch_size)
        ]

        # 每批单独 put：worker 只反序列化本批，避免「整表 + OSQP」打挂进程
        futures = []
        for batch in batches:
            batch_ref = ray.put(batch)
            minwt_ref = ray.put(minwt)
            mujieat_ref = ray.put(mujieat)
            futures.append(
                _getdtoptport_.remote(batch_ref, minwt_ref, mujieat_ref)
            )

        parts = ray.get(futures)
        funduse = pd.concat(parts, axis=0, ignore_index=True)
        if funduse.empty:
            raise ValueError("funduse 为空，无结果")

        funduse.columns = ["date", "Stkcd", "wt"]
        funduse["wt"] = funduse["wt"] / funduse.groupby("date")["wt"].transform("sum")
        return funduse, None, None
