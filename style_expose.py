"""
还原自研究员 Python：style_expose + getoptport 多日组合优化流程。

对应关系
--------
style_expose()
  - 组装 alpha / dot(风格+行业+sumticker) / mul(个股上下界)
  - 通用分支：bm_exp ± style_wt / industry_wt；个股 wt ± stock_wt（下界截到 0）
  - 不做 A 股特殊收紧分支

getoptport()
  - 按 date 循环，每日调用单日 LP（本文件默认走 scipy.linprog）
  - 失败则按 alpha 取前 1/mujieat 只等权兜底
  - 最后按日归一化权重

依赖：numpy / pandas / scipy（复用 optport_linprog.OptPortClassLinprog）
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from optport_linprog import OptPortClassLinprog


def _industry_one_hot(df: pd.DataFrame, industry_list: Sequence[str]) -> pd.DataFrame:
    """按固定行业列表做 one-hot，缺列补 0。"""
    out = df.copy()
    out["industry_name"] = pd.Categorical(
        out["industry_name"], categories=list(industry_list)
    )
    out = pd.get_dummies(
        out, columns=["industry_name"], prefix="", prefix_sep="", dtype=float
    )
    for ind in industry_list:
        if ind not in out.columns:
            out[ind] = 0.0
    out[list(industry_list)] = out[list(industry_list)].astype(float)
    return out


class StyleExposeOptimizer:
    """
    对照截图还原的 style_expose 封装。

    期望 metrics_df 列：
      date, wind_code, resid, wt, industry_name, + barra_names...
    """

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        barra_names: Sequence[str],
        industry_names: Sequence[str],
        style_wt: float = 0.15,
        industry_wt: float = 0.10,
        stock_wt: float = 0.05,
        factor_name: str = "resid",
        mujieat: float = 0.05,
        minwt: Optional[float] = None,
    ) -> None:
        self.metrics_df = metrics_df.copy()
        self.barra_names = list(barra_names)
        self.industry_names = list(industry_names)
        self.style_wt = float(style_wt)
        self.industry_wt = float(industry_wt)
        self.stock_wt = float(stock_wt)
        self.factor_name = factor_name
        self.mujieat = float(mujieat)
        self.minwt = minwt

    def style_expose(self) -> pd.DataFrame:
        """
        风格/行业暴露约束 + 个股上下界，再多日优化。

        Returns
        -------
        port_wt : DataFrame[date, Stkcd, wt]
        """
        lower_bond = 0.0
        upper_bond = 1.0
        industry_list = list(self.industry_names)

        def calc_bm_style(data: pd.DataFrame) -> pd.Series:
            cols = [*self.barra_names, *industry_list]
            return data[cols].mul(data["wt"], axis=0).sum()

        # ---- alpha ----
        alpha_final = self.metrics_df[["date", "wind_code", "resid"]].copy()

        # ---- 个股暴露矩阵 dot_mat：sumticker + 风格 + 行业 one-hot ----
        dot_mat = self.metrics_df[
            ["date", "wind_code", *self.barra_names, "industry_name"]
        ].copy()
        dot_mat["sumticker"] = 1.0
        dot_mat = dot_mat.fillna(0)
        dot_mat = _industry_one_hot(dot_mat, industry_list)

        # ---- 基准暴露 bm_exp（按日加权）----
        bm_exp = self.metrics_df[
            ["date", *self.barra_names, "wt", "industry_name"]
        ].copy()
        bm_exp = _industry_one_hot(bm_exp, industry_list)
        bm_exp = (
            bm_exp.groupby("date", group_keys=False)
            .apply(calc_bm_style)
            .reset_index()
        )

        # ---- 风格/行业上下界（通用分支）----
        lower_stock_exp = bm_exp[["date", *self.barra_names]].set_index("date")
        upper_stock_exp = bm_exp[["date", *self.barra_names]].set_index("date")
        lower_ind_exp = bm_exp[["date", *industry_list]].set_index("date")
        upper_ind_exp = bm_exp[["date", *industry_list]].set_index("date")

        lower_stock_exp = lower_stock_exp - self.style_wt
        upper_stock_exp = upper_stock_exp + self.style_wt
        lower_ind_exp = lower_ind_exp - self.industry_wt
        upper_ind_exp = upper_ind_exp + self.industry_wt

        # 行业界裁剪到 [0, 1]
        lower_ind_exp = lower_ind_exp.clip(lower=lower_bond, upper=None)
        upper_ind_exp = upper_ind_exp.clip(lower=None, upper=upper_bond)

        # ---- 个股上下界：mulmat=全1 ----
        mul_lower_bond = self.metrics_df[["date", "wind_code", "wt"]].copy()
        mul_upper_bond = self.metrics_df[["date", "wind_code", "wt"]].copy()
        mul_lower_bond = mul_lower_bond.rename(columns={"wt": "codelmtticker"})
        mul_upper_bond = mul_upper_bond.rename(columns={"wt": "codelmtticker"})

        mul_mat = self.metrics_df[["date", "wind_code"]].copy()
        mul_mat["codelmtticker"] = 1.0

        mul_lower_bond["codelmtticker"] = (
            mul_lower_bond["codelmtticker"] - self.stock_wt
        ).clip(lower=0.0)
        mul_upper_bond["codelmtticker"] = (
            mul_upper_bond["codelmtticker"] + self.stock_wt
        )

        # ---- 拼 dot 上下界，并加 sumticker=1（权重和=1）----
        dot_lower_bond = pd.concat([lower_stock_exp, lower_ind_exp], axis=1)
        dot_upper_bond = pd.concat([upper_stock_exp, upper_ind_exp], axis=1)
        dot_lower_bond["sumticker"] = 1.0
        dot_upper_bond["sumticker"] = 1.0

        # ---- 字段对齐研究员算法 ----
        alpha_final.columns = ["date", "Stkcd", self.factor_name]
        alpha_final = alpha_final.sort_values(["date", "Stkcd"]).reset_index(drop=True)

        factor_cols = ["sumticker", *self.barra_names, *industry_list]
        dot_mat = dot_mat.rename(columns={"wind_code": "Stkcd"})
        dot_mat = dot_mat[["date", "Stkcd", *factor_cols]]
        dot_mat = dot_mat.sort_values(["date", "Stkcd"]).set_index(["date", "Stkcd"])

        dot_lower_bond = dot_lower_bond.reindex(columns=factor_cols)
        dot_upper_bond = dot_upper_bond.reindex(columns=factor_cols)

        mul_mat = mul_mat.rename(columns={"wind_code": "Stkcd"})
        mul_mat = mul_mat[["date", "Stkcd", "codelmtticker"]].set_index(
            ["date", "Stkcd"]
        )
        mul_lower_bond = mul_lower_bond.rename(columns={"wind_code": "Stkcd"})
        mul_upper_bond = mul_upper_bond.rename(columns={"wind_code": "Stkcd"})
        mul_lower_bond = (
            mul_lower_bond[["date", "Stkcd", "codelmtticker"]]
            .sort_values(["date", "Stkcd"])
            .set_index(["date", "Stkcd"])
        )
        mul_upper_bond = (
            mul_upper_bond[["date", "Stkcd", "codelmtticker"]]
            .sort_values(["date", "Stkcd"])
            .set_index(["date", "Stkcd"])
        )

        # alpha 列名适配 getoptport_（date, code, alpha）
        alpha_for_opt = alpha_final.rename(
            columns={"Stkcd": "code", self.factor_name: "alpha"}
        )

        port_wt, _, _ = OptPortClassLinprog().getoptport_(
            datecodealpha=alpha_for_opt,
            dotmat=dot_mat,
            dotlowerbond=dot_lower_bond,
            dotupperbond=dot_upper_bond,
            mulmat=mul_mat,
            mullowerbond=mul_lower_bond,
            mulupperbond=mul_upper_bond,
            minwt=self.minwt,
            mujieat=self.mujieat,
        )
        return port_wt


def _demo() -> None:
    """两日假数据，验证多日通路。"""
    rows = []
    for dt in (pd.Timestamp("2023-12-20"), pd.Timestamp("2023-12-21")):
        rows.extend(
            [
                {
                    "date": dt,
                    "wind_code": "000001.SZ",
                    "industry_name": "银行",
                    "resid": 0.12,
                    "wt": 0.25,
                    "bp": 0.5,
                    "growth": 0.2,
                    "size": -0.1,
                },
                {
                    "date": dt,
                    "wind_code": "000002.SZ",
                    "industry_name": "银行",
                    "resid": 0.08,
                    "wt": 0.20,
                    "bp": -0.2,
                    "growth": 0.1,
                    "size": 0.3,
                },
                {
                    "date": dt,
                    "wind_code": "000003.SZ",
                    "industry_name": "医药",
                    "resid": -0.05,
                    "wt": 0.20,
                    "bp": 0.1,
                    "growth": -0.3,
                    "size": 0.2,
                },
                {
                    "date": dt,
                    "wind_code": "000004.SZ",
                    "industry_name": "医药",
                    "resid": 0.03,
                    "wt": 0.15,
                    "bp": 0.3,
                    "growth": 0.0,
                    "size": -0.2,
                },
                {
                    "date": dt,
                    "wind_code": "000005.SZ",
                    "industry_name": "计算机",
                    "resid": 0.10,
                    "wt": 0.20,
                    "bp": -0.4,
                    "growth": 0.4,
                    "size": 0.1,
                },
            ]
        )
    metrics = pd.DataFrame(rows)
    opt = StyleExposeOptimizer(
        metrics_df=metrics,
        barra_names=["bp", "growth", "size"],
        industry_names=["银行", "医药", "计算机"],
        style_wt=0.15,
        industry_wt=0.10,
        stock_wt=0.05,
        mujieat=0.05,
        minwt=None,
    )
    port = opt.style_expose()
    print(port)
    print(port.groupby("date")["wt"].sum())


if __name__ == "__main__":
    _demo()
