"""
原始 cvxpy/ECOS 版组合优化 + Ray 按批并行。

用途
----
与 optport_osqp.py 对照：验证「整表一次 put 进 worker」是否同样打挂 Ray，
以及「驱动切日 + 每批只 put 本批小数据」在 cvxpy 下是否稳定。

改造要点（与已验证可跑的 OSQP+Ray 版一致）
------------------------------------------
1. 驱动侧按日切片，不把完整多日大表塞进单个 remote
2. 日期按 batch_size（默认 100）切批，每批单独 ray.put
3. fallback / 噪声清理放进 remote 的 _getdtoptport_ 内执行
4. 返回 (funduse, None, None)，只关心权重

依赖
----
cvxpy / ecos（或 cvxpy 自带求解器）
numpy / pandas
ray（项目内已初始化：from ray.util.client import ray）

调用示例
--------
from optport_cvxpy_ray import OptPortClassCvxpyRay

opt = OptPortClassCvxpyRay()
port_wt, _, _ = opt.getoptport_(
    datecodealpha=alpha_df,
    dotmat=dat_mat,
    dotlowerbond=dot_lower,
    dotupperbond=dot_upper,
    mulmat=mul_mat,
    mullowerbond=mul_lower,
    mulupperbond=mul_upper,
    minwt=None,
    mujieat=0.05,
    batch_size=100,
)
"""

from __future__ import annotations

import gc
import logging
import warnings
from typing import Any, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from ray.util.client import ray

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 20  # 更稳：100 容易把 worker 打挂，默认改为 20
DEFAULT_MAX_INFLIGHT = 2  # 同时在飞的 Ray 批次数，降低 Client/内存压力


def _solve_one_day_cvxpy(
    codealpha: pd.DataFrame,
    dotmat: Optional[pd.DataFrame],
    dotlowerbond: Optional[pd.DataFrame],
    dotupperbond: Optional[pd.DataFrame],
    mulmat: Optional[pd.DataFrame],
    mullowerbond: Optional[pd.DataFrame],
    mulupperbond: Optional[pd.DataFrame],
    minwt: Optional[float],
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    单日 cvxpy/ECOS 求解（对齐原始 _getdtoptport_ / getobj_ 语义）。

    目标：maximize weight.T @ alpha
    约束：
      - dot:  dotlowerbond <= weight.T @ dotmat <= dotupperbond
      - mul:  mullowerbond <= weight * mulmat <= mulupperbond
    """
    codealpha_idx = codealpha.set_index("code")
    N = int(codealpha_idx.shape[0])
    if N <= 0:
        return None, None

    # 原始：self.weight = cp.Variable(shape=(N, 1))
    weight = cp.Variable(shape=(N, 1))

    # 原始 getobj_：portalpha = self.weight.T @ self.codealpha
    alpha_col = codealpha_idx[["alpha"]].astype(float).fillna(0.0)
    # 用 numpy 值进入表达式，避免 DataFrame 在 worker 上的边界问题
    alpha_mat = alpha_col.to_numpy(dtype=float)
    objective = cp.Maximize(weight.T @ alpha_mat)

    constraints = []

    # dot 约束：weight.T @ dotmat  ∈ [lower, upper]
    if dotmat is not None and dotlowerbond is not None and dotupperbond is not None:
        dm = np.nan_to_num(dotmat.to_numpy(dtype=float), nan=0.0)  # (N, F)
        dl = np.nan_to_num(
            dotlowerbond.to_numpy(dtype=float).flatten(), nan=-1e20
        ).reshape(1, -1)
        du = np.nan_to_num(
            dotupperbond.to_numpy(dtype=float).flatten(), nan=1e20
        ).reshape(1, -1)
        dotres = weight.T @ dm  # (1, F)
        constraints.append(dotres >= dl)
        constraints.append(dotres <= du)

    # mul 约束：cp.multiply(weight, mulmat) ∈ [lower, upper]
    if mulmat is not None and mullowerbond is not None and mulupperbond is not None:
        mv = np.nan_to_num(mulmat.to_numpy(dtype=float).flatten(), nan=1.0).reshape(-1, 1)
        ml = np.nan_to_num(
            mullowerbond.to_numpy(dtype=float).flatten(), nan=-1e20
        ).reshape(-1, 1)
        mu = np.nan_to_num(
            mulupperbond.to_numpy(dtype=float).flatten(), nan=1e20
        ).reshape(-1, 1)
        mulres = cp.multiply(weight, mv)
        constraints.append(mulres >= ml)
        constraints.append(mulres <= mu)

    prob = cp.Problem(objective, constraints)

    try:
        # 原始代码使用 ECOS；屏蔽 cvxpy 关于 1.6.0 默认不再附带 ECOS 的 FutureWarning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*ECOS will no longer be installed by default.*",
                category=FutureWarning,
            )
            prob.solve(verbose=False, solver=cp.ECOS)
    except Exception as e:
        logger.error(f"cvxpy/ECOS 求解异常: {e}")
        return None, None

    status = prob.status
    if weight.value is None or status not in ("optimal", "optimal_inaccurate"):
        logger.warning(f"cvxpy 未求解，状态: {status}")
        return None, status

    # ── 原始后处理（与旧 _getdtoptport_ 对齐）────────────────────────
    tmp_weight = pd.Series(
        np.asarray(weight.value).flatten(),
        index=codealpha_idx.index,
        name="wt",
    ).to_frame()

    tmp_weight[tmp_weight <= 0] = np.nan

    if (
        (tmp_weight["wt"] <= 1e-5).all()
        or np.isinf(tmp_weight.values).any()
        or tmp_weight["wt"].isna().all()
        or (tmp_weight["wt"].max() > 1e20)
    ):
        logger.warning("cvxpy 解异常（全零/inf/nan/过大），判定失败")
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

    # 尽快丢掉 cvxpy 问题对象，减轻 worker 内存碎片
    del weight, objective, constraints, prob, alpha_mat
    gc.collect()
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
    """驱动侧按日切片（与原始 getoptport_ 循环内逻辑一致）。"""
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

        mullowerbonddt = mullowerbond.loc[[optdt]].reset_index(
            level="date", drop=True
        )
        mullowerbonddt = mullowerbonddt[~mullowerbonddt.index.duplicated()].reindex(
            codes
        )

        mulupperbonddt = mulupperbond.loc[[optdt]].reset_index(
            level="date", drop=True
        )
        mulupperbonddt = mulupperbonddt[~mulupperbonddt.index.duplicated()].reindex(
            codes
        )

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
    Ray remote：一批已切好的交易日。

    内含：cvxpy 求解 + 失败 fallback + 噪声清理。
    返回 columns=[date, code, wt]。
    """
    portwtlist: list[pd.DataFrame] = []

    for day in batch_days:
        optdt = day["optdt"]
        codealpha = day["codealpha"]

        portwttemp, _status = _solve_one_day_cvxpy(
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
            # 优化失败：按 alpha 降序等权选股（原始 getoptport_ 逻辑）
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
            # 与原始 getoptport_ 一致的噪声清理（删行）
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

        # 释放本日本地大对象，降低批内峰值内存
        del portwttemp, codealpha
        gc.collect()

    if not portwtlist:
        return pd.DataFrame(columns=["date", "code", "wt"])
    return pd.concat(portwtlist, axis=0, ignore_index=True)


class OptPortClassCvxpyRay:
    """原始 cvxpy/ECOS 语义 + Ray 按批并行（按需 put 本批切片）。"""

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
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[float]]:
        """
        驱动切日 → 每批 put 本批小数据 → 限制并发提交 Ray → 汇总归一。

        Parameters
        ----------
        batch_size : 每批交易日数（默认 20，比 100 更稳）
        max_inflight : 同时在飞批次数（默认 2），避免一次打满 Client/worker

        Returns
        -------
        (port_wt, None, None)
        """
        if batch_size <= 0:
            raise ValueError("batch_size 必须 > 0")
        if max_inflight <= 0:
            raise ValueError("max_inflight 必须 > 0")

        datecodealpha = datecodealpha.copy()
        if isinstance(datecodealpha.index, pd.MultiIndex):
            datecodealpha = datecodealpha.reset_index()
        datecodealpha.columns = ["date", "code", "alpha"]

        optperiods = sorted(datecodealpha["date"].unique())

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

        # 标量只 put 一次
        minwt_ref = ray.put(minwt)
        mujieat_ref = ray.put(mujieat)

        parts: list[pd.DataFrame] = []
        pending: list[Any] = []
        batch_iter = iter(batches)

        def _submit_one(batch: list[dict[str, Any]]) -> Any:
            batch_ref = ray.put(batch)
            return _getdtoptport_.remote(batch_ref, minwt_ref, mujieat_ref)

        try:
            # 先填满 in-flight 窗口
            for _ in range(min(max_inflight, len(batches))):
                try:
                    pending.append(_submit_one(next(batch_iter)))
                except StopIteration:
                    break

            while pending:
                # 取回已完成的，再补提交，控制并发
                ready, pending = ray.wait(pending, num_returns=1)
                parts.append(ray.get(ready[0]))
                try:
                    pending.append(_submit_one(next(batch_iter)))
                except StopIteration:
                    pass

        except Exception as e:
            # Ray Client 会话一旦因 worker 崩溃被清理，后续 put/get 都会失败
            err = str(e)
            if (
                "already been disconnected" in err
                or "cleaned up" in err
                or "ConnectionError" in type(e).__name__
            ):
                raise RuntimeError(
                    "Ray Client 会话已断开（常见原因：worker 上 cvxpy/ECOS 崩溃或 OOM）。"
                    "请先重连/重启 Ray，再重试；可把 batch_size=10、max_inflight=1 再试。"
                ) from e
            raise

        funduse = pd.concat(parts, axis=0, ignore_index=True)
        if funduse.empty:
            raise ValueError("funduse 为空，无结果")

        funduse.columns = ["date", "Stkcd", "wt"]
        funduse["wt"] = funduse["wt"] / funduse.groupby("date")["wt"].transform("sum")
        return funduse, None, None
