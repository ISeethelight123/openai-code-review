"""
Portfolio optimization with scipy.optimize.linprog (no cvxpy / osqp / ray).

问题形式（与原始 cvxpy 版一致）
--------------------------------
maximize   alpha^T @ w
s.t.       dotlower <= w^T @ dotmat <= dotupper
           mullower <= w * mulmat   <= mulupper
           w >= 0

scipy.linprog 为 minimize，故使用 c = -alpha。

依赖（项目已有）
---------------
numpy / pandas / scipy

调用示例
--------
from optport_linprog import OptPortClassLinprog

opt = OptPortClassLinprog()
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
)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

# linprog 不接受 ±inf，用有限大数代替
_LINPROG_INF = 1e20


def _finite(arr: np.ndarray, fill: float) -> np.ndarray:
    """把 nan/±inf 收成有限值。"""
    out = np.asarray(arr, dtype=np.float64).flatten()
    out = np.nan_to_num(out, nan=fill, posinf=_LINPROG_INF, neginf=-_LINPROG_INF)
    return out


class OptPortClassLinprog:
    """
    组合优化类（scipy.linprog / HiGHS）。

    对外：
      - getoptport_    : 按日串行求解并汇总
      - _getdtoptport_ : 单日 LP 求解
    """

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
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[float]]:
        """
        按日期串行调用 _getdtoptport_，汇总结果。

        Returns
        -------
        (port_wt, None, None)
          port_wt : columns=[date, Stkcd, wt]
        """
        datecodealpha = datecodealpha.copy()
        if isinstance(datecodealpha.index, pd.MultiIndex):
            datecodealpha = datecodealpha.reset_index()
        datecodealpha.columns = ["date", "code", "alpha"]

        optperiods = sorted(datecodealpha["date"].unique())
        portwtlist: list[pd.DataFrame] = []

        for optdt in optperiods:
            codealpha = (
                datecodealpha.loc[datecodealpha["date"] == optdt, ["code", "alpha"]]
                .drop_duplicates("code")
                .copy()
            )
            codes = sorted(codealpha["code"].tolist())
            # 与 dot/mul 矩阵行顺序严格对齐（按 code 排序）
            codealpha = (
                codealpha.set_index("code")
                .reindex(codes)
                .reset_index()
            )
            codealpha["alpha"] = pd.to_numeric(
                codealpha["alpha"], errors="coerce"
            ).fillna(0.0)
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
                mullowerbonddt = mullowerbonddt[
                    ~mullowerbonddt.index.duplicated()
                ].reindex(codes)

                mulupperbonddt = mulupperbond.loc[[optdt]].reset_index(
                    level="date", drop=True
                )
                mulupperbonddt = mulupperbonddt[
                    ~mulupperbonddt.index.duplicated()
                ].reindex(codes)

            portwttemp, _status = self._getdtoptport_(
                codealpha=codealpha,
                dotmat=dotmatdt,
                dotlowerbond=dotlowerbonddt,
                dotupperbond=dotupperbonddt,
                mulmat=mulmatdt,
                mullowerbond=mullowerbonddt,
                mulupperbond=mulupperbonddt,
                minwt=minwt,
            )

            if portwttemp is None:
                # 优化失败：按 alpha 降序等权选股
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
            raise ValueError("计算结果为空：portwtlist 为空")

        funduse = pd.concat(portwtlist, axis=0, ignore_index=True)
        if funduse.empty:
            raise ValueError("funduse 为空，无结果")

        funduse.columns = ["date", "Stkcd", "wt"]
        funduse["wt"] = funduse["wt"] / funduse.groupby("date")["wt"].transform("sum")
        return funduse, None, None

    def _getdtoptport_(
        self,
        codealpha: pd.DataFrame,
        dotmat: Optional[pd.DataFrame] = None,
        dotlowerbond: Optional[pd.DataFrame] = None,
        dotupperbond: Optional[pd.DataFrame] = None,
        mulmat: Optional[pd.DataFrame] = None,
        mullowerbond: Optional[pd.DataFrame] = None,
        mulupperbond: Optional[pd.DataFrame] = None,
        minwt: Optional[float] = None,
        currentWeight: Optional[np.ndarray] = None,
    ) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        单日 LP：scipy.optimize.linprog。

        maximize alpha^T w  <=>  minimize (-alpha)^T w
        s.t.  A_ub @ w <= b_ub
              0 <= w
        """
        codealpha_idx = codealpha.set_index("code")
        N = int(codealpha_idx.shape[0])
        if N <= 0:
            return None, None

        alpha = _finite(codealpha_idx["alpha"].to_numpy(), fill=0.0)
        c = -alpha  # maximize -> minimize

        A_list: list[np.ndarray] = []
        b_list: list[np.ndarray] = []

        # 1) dot: dl <= Dot^T w <= du
        #    Dot 为 (N, F)，约束写成：
        #      Dot^T w <= du
        #     -Dot^T w <= -dl
        if dotmat is not None and dotlowerbond is not None and dotupperbond is not None:
            dm = np.nan_to_num(dotmat.to_numpy(dtype=np.float64), nan=0.0)  # (N, F)
            if dm.ndim == 1:
                dm = dm.reshape(-1, 1)
            a_dot = dm.T  # (F, N)
            dl = _finite(dotlowerbond.to_numpy(), fill=-_LINPROG_INF)
            du = _finite(dotupperbond.to_numpy(), fill=_LINPROG_INF)
            if dl.size != a_dot.shape[0] or du.size != a_dot.shape[0]:
                logger.warning(
                    f"dot 约束维度不匹配: F={a_dot.shape[0]}, "
                    f"dl={dl.size}, du={du.size}"
                )
                return None, "dim_mismatch_dot"
            A_list.append(a_dot)
            b_list.append(du)
            A_list.append(-a_dot)
            b_list.append(-dl)

        # 2) mul: ml <= m ⊙ w <= mu
        #      diag(m) w <= mu
        #     -diag(m) w <= -ml
        if mulmat is not None and mullowerbond is not None and mulupperbond is not None:
            mv = _finite(mulmat.to_numpy(), fill=1.0)
            ml = _finite(mullowerbond.to_numpy(), fill=-_LINPROG_INF)
            mu = _finite(mulupperbond.to_numpy(), fill=_LINPROG_INF)
            if mv.size != N or ml.size != N or mu.size != N:
                logger.warning(
                    f"mul 约束维度不匹配: N={N}, m={mv.size}, "
                    f"ml={ml.size}, mu={mu.size}"
                )
                return None, "dim_mismatch_mul"
            a_mul = np.diag(mv)
            A_list.append(a_mul)
            b_list.append(mu)
            A_list.append(-a_mul)
            b_list.append(-ml)

        A_ub = np.vstack(A_list) if A_list else None
        b_ub = np.concatenate(b_list) if b_list else None

        # 与组合实务一致：长期只（后处理也会丢掉负权重）
        bounds = [(0.0, None)] * N

        try:
            # HiGHS 是 scipy>=1.6 推荐 LP 求解器；1.10 可用
            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs",
                options={"presolve": True, "time_limit": 30.0},
            )
        except Exception as e:
            logger.error(f"linprog 求解异常: {e}")
            return None, None

        status = result.message if result.message else str(result.status)
        if (not result.success) or result.x is None:
            logger.warning(f"linprog 未求解成功: status={result.status}, msg={status}")
            return None, status

        # ── 与旧 cvxpy 版对齐的后处理 ─────────────────────────────────
        tmp_weight = pd.Series(
            np.asarray(result.x, dtype=np.float64).flatten(),
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
            logger.warning("linprog 解异常（全零/inf/nan/过大），判定失败")
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
