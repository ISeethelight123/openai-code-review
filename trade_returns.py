"""
根据买入价 / 卖出价枚举计算日、周、月收益率。

两条独立入口（互不调用）：
  - calc_trade_returns：宽表（index=日期, columns=code）
  - calc_trade_returns_long：长表（date, code, 价格列）

日收益：先按枚举 offset 取买卖价，再 r = P_sell / P_buy - 1。
周 / 月：按股票在 ISO 周 / 自然月内跳过 NaN 连乘；
        只在该股该期最后一个有行情日赋值，其余为 NaN。
"""

from __future__ import annotations

from enum import Enum
from typing import Final

import numpy as np
import pandas as pd


class PriceType(Enum):
    """价格类型。"""

    OPEN = "open"
    CLOSE = "close"
    VWAP = "vwap"
    TWAP = "twap"  # 暂无数据，调用时会报错


# 未复权价格字段映射
PRICE_COL_MAP: Final[dict[PriceType, str]] = {
    PriceType.OPEN: "s_dq_open",
    PriceType.CLOSE: "s_dq_close",
    PriceType.VWAP: "s_dq_avgprice",
}


class TradePrice(Enum):
    """
    买卖价枚举：每个值 = (相对交易日偏移, 价格类型)。

    T 日 = 当日（offset=0），T+1 = 下一交易日（offset=1），以此类推。
    """

    T_CLOSE = (0, PriceType.CLOSE)
    T1_OPEN = (1, PriceType.OPEN)
    T1_CLOSE = (1, PriceType.CLOSE)
    T1_VWAP = (1, PriceType.VWAP)
    T1_TWAP = (1, PriceType.TWAP)
    T2_OPEN = (2, PriceType.OPEN)
    T2_VWAP = (2, PriceType.VWAP)
    T2_TWAP = (2, PriceType.TWAP)

    @property
    def offset(self) -> int:
        return self.value[0]

    @property
    def price_type(self) -> PriceType:
        return self.value[1]


def _resolve_price_column(price: TradePrice) -> str:
    """解析枚举对应的价格列名；TWAP 未实现。"""
    if price.price_type is PriceType.TWAP:
        raise NotImplementedError("TWAP 暂无数据，尚未实现")
    return PRICE_COL_MAP[price.price_type]


def _period_end_compound_panel(
    ret: pd.DataFrame,
    freq: str,
    active: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """宽表：自然期内按列复合，仅在该列该期最后一个有行情日赋值。"""
    periods = ret.index.to_period(freq)
    factor = (1.0 + ret).fillna(1.0)
    n_valid = ret.notna().groupby(periods).transform("sum")
    compound = factor.groupby(periods).transform("prod") - 1.0
    compound = compound.where(n_valid > 0)

    # active：有行情的日期；默认用 ret 非空（兼容）。调用方应传入价格非空掩码。
    if active is None:
        active = ret.notna()
    date_panel = pd.DataFrame(
        np.broadcast_to(ret.index.to_numpy()[:, None], ret.shape),
        index=ret.index,
        columns=ret.columns,
    ).where(active)
    last_date = date_panel.groupby(periods).transform("max")
    is_end = active & date_panel.eq(last_date)
    return compound.where(is_end)


def _period_end_compound_long(
    codes: pd.Series,
    dates: pd.Series,
    ret: pd.Series,
    period: pd.Series,
) -> pd.Series:
    """长表：自然期内复合，仅在每个 (code, period) 最后一个交易日赋值。"""
    keys = [codes, period]
    factor = (1.0 + ret).fillna(1.0)
    n_valid = ret.notna().groupby(keys, sort=False).transform("sum")
    compound = factor.groupby(keys, sort=False).transform("prod") - 1.0
    compound = compound.where(n_valid > 0)
    is_end = dates == dates.groupby(keys, sort=False).transform("max")
    return compound.where(is_end)


def calc_trade_returns(
    buy_px: pd.DataFrame,
    sell_px: pd.DataFrame,
    buy_price: TradePrice,
    sell_price: TradePrice,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    宽表入口：买卖价面板 → (ret_daily, ret_weekly, ret_monthly)。

    buy_px / sell_px：index=日期，columns=code，值为对应价格（未做偏移）。
    日期可为 datetime.date / datetime / 可解析字符串。
    """
    if (
        buy_price.price_type is PriceType.TWAP
        or sell_price.price_type is PriceType.TWAP
    ):
        raise NotImplementedError("TWAP 暂无数据，尚未实现")

    buy_px = buy_px.copy()
    sell_px = sell_px.copy()
    buy_px.index = pd.to_datetime(buy_px.index)
    sell_px.index = pd.to_datetime(sell_px.index)

    buy_px, sell_px = buy_px.align(sell_px, join="inner")
    buy_px = buy_px.sort_index()
    sell_px = sell_px.reindex_like(buy_px)

    buy = buy_px.shift(-buy_price.offset)
    sell = sell_px.shift(-sell_price.offset)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret_daily = sell / buy - 1.0
    ret_daily = ret_daily.mask(~np.isfinite(ret_daily))

    # 有行情日：原始买入价非空（与长表「该股有行」对齐）
    active = buy_px.notna()
    ret_weekly = _period_end_compound_panel(ret_daily, "W-SUN", active=active)
    ret_monthly = _period_end_compound_panel(ret_daily, "M", active=active)
    return ret_daily, ret_weekly, ret_monthly


def calc_trade_returns_long(
    df: pd.DataFrame,
    buy_price: TradePrice,
    sell_price: TradePrice,
    *,
    date_col: str = "trade_date",
    code_col: str = "code",
) -> pd.DataFrame:
    """
    长表入口：date + code + 价格列 → 增加 ret_daily / ret_weekly / ret_monthly。

    独立实现，不调用宽表函数。
    算出日收益后丢弃无效行（空 / inf），再在剩余样本上算周 / 月。
    """
    buy_col = _resolve_price_column(buy_price)
    sell_col = _resolve_price_column(sell_price)

    required = {date_col, code_col, buy_col, sell_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"缺少必要列: {sorted(missing)}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([code_col, date_col], kind="mergesort").reset_index(drop=True)

    grouped = out.groupby(code_col, sort=False)
    buy_px = grouped[buy_col].shift(-buy_price.offset)
    sell_px = grouped[sell_col].shift(-sell_price.offset)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["ret_daily"] = sell_px / buy_px - 1.0

    # 只保留有效日收益，再算周 / 月（期末 = 剩余行中该股该期最后一天）
    out = out.loc[np.isfinite(out["ret_daily"].to_numpy())].reset_index(drop=True)

    dates = out[date_col]
    codes = out[code_col]
    ret_daily = out["ret_daily"]
    out["ret_weekly"] = _period_end_compound_long(
        codes, dates, ret_daily, dates.dt.to_period("W-SUN")
    )
    out["ret_monthly"] = _period_end_compound_long(
        codes, dates, ret_daily, dates.dt.to_period("M")
    )
    return out


if __name__ == "__main__":
    import datetime as dt

    idx = [
        dt.date(2021, 1, 4),
        dt.date(2021, 1, 5),
        dt.date(2021, 1, 6),
        dt.date(2021, 1, 7),
        dt.date(2021, 1, 8),
    ]
    open_px = pd.DataFrame(
        {
            "000001": [10.0, 10.2, 10.1, 10.3, 10.5],
            "000002": [20.0, 20.5, 21.0, np.nan, np.nan],
        },
        index=idx,
    )

    ret_d, ret_w, ret_m = calc_trade_returns(
        open_px,
        open_px,
        buy_price=TradePrice.T1_OPEN,
        sell_price=TradePrice.T2_OPEN,
    )
    print("===== wide =====")
    print(ret_d)
    print(ret_w)

    demo_long = pd.DataFrame(
        {
            "trade_date": idx + idx[:3],
            "code": ["000001"] * 5 + ["000002"] * 3,
            "s_dq_open": [10.0, 10.2, 10.1, 10.3, 10.5, 20.0, 20.5, 21.0],
        }
    )
    print("===== long =====")
    print(
        calc_trade_returns_long(
            demo_long,
            buy_price=TradePrice.T1_OPEN,
            sell_price=TradePrice.T2_OPEN,
        )[["trade_date", "code", "ret_daily", "ret_weekly", "ret_monthly"]]
    )
