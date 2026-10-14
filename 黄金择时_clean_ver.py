

import os
import numpy as np
import pandas as pd

os.chdir(r'D:\JYH\AAA深度报告\黄金')


end_date = '2026-06-04'


# ----------定期信号-----------


# 官方黄金储备

gold_storage_data = pd.read_excel('黄金择时原始数据_clean_ver.xlsx', sheet_name='黄金储备')
gold_storage_data.columns = 'DATE','storage1','storage2'
gold_storage_data = gold_storage_data.iloc[2:].set_index('DATE')
gold_storage_data.index = pd.to_datetime(gold_storage_data.index)

# 2018年前-原始数据是季度数据
gold_storage_q = gold_storage_data.storage1.loc[:'2017-12']
# 2018年后-原始数据是月度数据，windEDB已取季度末值
gold_storage_m = gold_storage_data.storage2.loc['2018-01':]

## 季频
gold_storage_q_all = pd.DataFrame(pd.concat([gold_storage_q, gold_storage_m])).astype('float').rename(columns={0:'GOLD_STORAGE'})
gold_storage_q_all['STORAGE_DIFF'] = gold_storage_q_all.GOLD_STORAGE.diff()
gold_storage_signal = (gold_storage_q_all.STORAGE_DIFF > 65).ffill().dropna().astype('float')
gold_storage_signal = gold_storage_signal.loc['2011-12':end_date]



# 黄金ETF资金流入

gold_etf_hold = pd.read_excel('黄金择时原始数据_clean_ver.xlsx', sheet_name='黄金ETF')
gold_etf_hold.columns = 'DATE','SPDR','ISHARE'
gold_etf_hold = gold_etf_hold.iloc[2:].set_index('DATE')
gold_etf_hold.index = pd.to_datetime(gold_etf_hold.index)
gold_etf_hold = gold_etf_hold.dropna()

## 月频
gold_etf_hold_m = gold_etf_hold.resample('M').last()
gold_etf_hold_diff_m = gold_etf_hold_m.diff().dropna()
gold_etf_hold_signal = (gold_etf_hold_diff_m.SPDR>0)&(gold_etf_hold_diff_m.ISHARE>0)
gold_etf_hold_signal = gold_etf_hold_signal.dropna().astype('float').loc['2011-12':end_date]



# EPU

daily_risk_data = pd.read_excel('黄金择时原始数据_clean_ver.xlsx', sheet_name='EPU')
daily_risk_data.columns = 'DATE','EPU'
daily_risk_data = daily_risk_data.iloc[2:].set_index('DATE')
daily_risk_data.index = pd.to_datetime(daily_risk_data.index)

risk_data = daily_risk_data.resample('W').mean()
risk_data['EPU_PCT'] = np.nan

for ind in risk_data.index: 
    epu_data = risk_data.loc[:ind, 'EPU']
    risk_data.loc[ind,'EPU_PCT'] = (epu_data < epu_data[-1]).sum() / len(epu_data)

risk_wdata = risk_data.resample('W').last()
epu_signal = (risk_wdata.EPU_PCT > 0.9).astype('float').loc['2011-12-25':end_date]




# 定期信号全部转化为日频

# gold price
close = pd.read_excel('黄金择时原始数据_clean_ver.xlsx', sheet_name='金价').dropna()
close = close.set_index('DATE')
close.index = pd.to_datetime(close.index)
close = close.rename(columns={'SPTAUUSDOZ.IDC':'GOLD'})

def indc_to_daily(df_signal, df_close, freq, start_date, end_date):
    
    # 转为日频信号
    trad_date = pd.DataFrame(df_close.index)
    trad_date.index = trad_date.iloc[:,0]
    trad_date = trad_date.resample(freq).last()
    trad_date = trad_date.reindex(index=df_signal.index)
    
    signal = df_signal.copy()
    signal.index = trad_date.iloc[:,0]
    
    daily_signal = signal.reindex(index=df_close.index).shift(1).ffill().loc[start_date:end_date].fillna(0)
   
    return daily_signal

gold_storage_holding = indc_to_daily(gold_storage_signal, close.GOLD, 'Q', '2011-12-30', end_date)
gold_etf_holding = indc_to_daily(gold_etf_hold_signal, close.GOLD, 'M', '2011-12-30', end_date)
epu_holding = indc_to_daily(epu_signal, close.GOLD, 'W', '2011-12-30', end_date)

# 全部定期信号
select_holding1 = pd.concat([gold_storage_holding, gold_etf_holding, epu_holding], axis=1)
select_holding1.columns = 'gold_storage','gold_etf','epu'



# ----------不定期信号-----------


# 期现基差

arbitrage_data = pd.read_excel('黄金择时原始数据_clean_ver.xlsx', sheet_name='期现基差')
arbitrage_data.columns = 'DATE','FUTURE_PRICE','GOLD_PRICE','COMEX_STORAGE'
arbitrage_data = arbitrage_data.iloc[2:].set_index('DATE')
arbitrage_data.index = pd.to_datetime(arbitrage_data.index)

arbitrage_data['basis'] = arbitrage_data.FUTURE_PRICE - arbitrage_data.GOLD_PRICE
arbitrage_data['basis_ma60'] = arbitrage_data.basis.rolling(60).mean()
arbitrage_data['basis_ma60_pct'] = arbitrage_data['basis_ma60'].rolling(250*3).apply(lambda x: (x<x[-1]).sum()/len(x))

arbitrage_signal = pd.DataFrame(arbitrage_data['basis_ma60_pct'].dropna())
arbitrage_signal['stage'] = np.nan
arbitrage_signal['stage'][(arbitrage_signal.basis_ma60_pct>0.9)&(arbitrage_signal.basis_ma60_pct.shift(1)<0.9)] = 1
arbitrage_signal['stage'][(arbitrage_signal.basis_ma60_pct<0.5)&(arbitrage_signal.basis_ma60_pct.shift(1)>0.5)] = 0
arbitrage_signal['stage'] = arbitrage_signal['stage'].ffill().shift(1).fillna(0)
arbitrage_signal = arbitrage_signal.reindex(index=select_holding1.index).ffill()



# 隐波趋势

# 隐波
iv_data = pd.read_excel('黄金择时原始数据_clean_ver.xlsx', sheet_name='隐波')
iv_data.columns = 'DATE','GVZ.GI','GOLD_PRICE'
iv_data = iv_data.iloc[2:].set_index('DATE')
iv_data.index = pd.to_datetime(iv_data.index)
iv_data = iv_data.ffill().dropna()

# 滚动3年分位数
iv_data['IV_PCT'] = iv_data['GVZ.GI'].rolling(250*3).apply(lambda x: (x<x[-1]).sum()/len(x))
iv_pct_signal = pd.DataFrame(iv_data['IV_PCT']).dropna()
iv_pct_signal['stage'] = np.nan

# 0: high vol; 0.5: med vol; 1: low vol
iv_pct_signal['stage'][(iv_pct_signal.IV_PCT>0.95)&(iv_pct_signal.IV_PCT.shift(1)<0.95)] = 0
iv_pct_signal['stage'][(iv_pct_signal.IV_PCT<0.8)&(iv_pct_signal.IV_PCT.shift(1)>0.8)] = 0.5
iv_pct_signal['stage'][(iv_pct_signal.IV_PCT<0.2)&(iv_pct_signal.IV_PCT.shift(1)>0.2)] = 1
iv_pct_signal['stage'][(iv_pct_signal.IV_PCT>0.5)&(iv_pct_signal.IV_PCT.shift(1)<0.5)] = 0.5
iv_pct_signal['stage'] = iv_pct_signal['stage'].ffill().shift(1).fillna(1)

# 趋势
window1 = 20
window2 = 60

trend_signal = pd.DataFrame(iv_data['GOLD_PRICE']).dropna()
trend_signal['MA1'] = trend_signal['GOLD_PRICE'].rolling(window1).mean()
trend_signal['MA2'] = trend_signal['GOLD_PRICE'].rolling(window2).mean()

trend_signal['stage'] = np.nan
trend_signal['stage'][(trend_signal.MA1>trend_signal.MA2)&
                      (trend_signal.shift(1).MA1<trend_signal.shift(1).MA2)] = 1
trend_signal['stage'][(trend_signal.MA1<trend_signal.MA2)&
                      (trend_signal.shift(1).MA1>trend_signal.shift(1).MA2)] = 0
trend_signal['stage'] = trend_signal['stage'].shift(1).ffill()
trend_signal = trend_signal.dropna().reindex(index=iv_pct_signal.index)

# 隐波+趋势
trend_iv_signal_output = pd.Series(index=iv_pct_signal.index)
trend_iv_signal_output[(trend_signal.stage == 1)&(iv_pct_signal.stage==1)] = 1
trend_iv_signal_output[(trend_signal.stage == 1)&(iv_pct_signal.stage==0.5)] = 1
trend_iv_signal_output = trend_iv_signal_output.fillna(0)
trend_iv_signal_output = trend_iv_signal_output.reindex(index=select_holding1.index).ffill()



# 不定期信号
select_holding2 = pd.concat([trend_iv_signal_output, arbitrage_signal['stage']], axis=1)
select_holding2.columns = 'iv_trend','basis'



# 全部信号
select_holding = pd.concat([select_holding1, select_holding2], axis=1)
select_holding.iloc[0] = 0




# 策略组合

## 黄金储备+黄金ETF+EPU
strategy_holding1 = pd.Series(index=select_holding.index)
strategy_holding1[select_holding.gold_storage == 0] = 0
strategy_holding1[(select_holding.gold_storage == 1)&
                  (select_holding.gold_etf == 0)&
                  (select_holding.epu == 0)] = 0.5
strategy_holding1[(select_holding.gold_storage == 1)&
                  ((select_holding.gold_etf == 1)|
                   (select_holding.epu == 1))] = 1

gld_dret = close.GOLD.pct_change(fill_method='pad').reindex(index = strategy_holding1.index).fillna(0)
strategy_nw1 = (1 + gld_dret*strategy_holding1).cumprod()

## 期现基差+隐波趋势
strategy_holding2 = pd.Series(index=select_holding.index)
strategy_holding2[(select_holding.iv_trend==0)&(select_holding.basis==0)] = 0
strategy_holding2[(select_holding.iv_trend==1)|(select_holding.basis==1)] = 1

strategy_nw2 = (1 + gld_dret*strategy_holding2).cumprod()


## 60/40
strategy_holding3 = 0.6*strategy_holding1 + 0.4*strategy_holding2
strategy_nw3 = (1 + gld_dret*strategy_holding3).cumprod()


def PorfolioPerformance(port_nw):
    
    # 年化表现
    port_nw = port_nw.astype('float')
    yr = (port_nw.index[-1] - port_nw.index[0]).days/365
    
    dret = port_nw.pct_change()
    anvol = dret.std()*np.sqrt(250)
    anret = pow(port_nw.iloc[-1],1/yr) - 1
    ## 最大回撤
    drawdown = 1 - port_nw / port_nw.cummax()
    max_drawdown = drawdown.max()
    ## 月度胜率
    mret = port_nw.resample('M').last().pct_change().dropna()
    winner_rate = (mret>0).sum() / len(mret)
   
    # 逐年区间表现
    # 区间收益
    year_ret = port_nw.resample('Y').last().pct_change()
    year_ret.iloc[0] = port_nw.resample('Y').last().iloc[0]-1
    
    # 区间波动
    yr_len = port_nw.resample('Y').count()
    year_vol = dret.resample('Y').std() * np.sqrt(yr_len)

    # 区间最大回撤
    year_maxdrawdown = port_nw.resample('Y').apply(lambda x: (1-x/x.cummax()).max())
       
    # 区间月度胜率
    year_winner_rate = mret.resample('Y').apply(lambda x: (x>0).sum() / len(x))
    
    # output
    output = pd.concat([year_ret, year_vol, year_maxdrawdown, year_winner_rate],axis=1)
    output.columns = 'ret', 'vol', 'dw', 'winner_rate'
    output.loc['annual'] = anret, anvol, max_drawdown, winner_rate
    output = output.dropna()
    ## 收益波动比
    output['ret_vol'] = output.ret/output.vol
    ## 收益回撤比
    output['ret_dw'] = output.ret/output.dw
   
    return output


all_holding = pd.concat([strategy_holding1, strategy_holding2, strategy_holding3], axis=1)
all_nw = pd.concat([strategy_nw1, strategy_nw2, strategy_nw3], axis=1)

all_holding.columns = 'regular','irregular','60_40'
all_nw.columns = all_holding.columns

gld_dret.iloc[0] = 0
benchmark_nw = (1 + gld_dret*0.5).cumprod()
all_nw['benchamrk_half'] = benchmark_nw

all_nw['ex_regular'] = all_nw.regular /  all_nw.benchamrk_half
all_nw['ex_irregular'] = all_nw.irregular /  all_nw.benchamrk_half
all_nw['ex_60_40'] = all_nw['60_40'] /  all_nw.benchamrk_half


for col in all_nw.columns:
    perf_i = PorfolioPerformance(all_nw[col])
    perf_i.to_excel(r'output\定期更新\perf_'+str(col)+'.xlsx')

select_holding.to_excel(r'output\定期更新\单信号.xlsx')
all_holding.to_excel(r'output\定期更新\组合仓位.xlsx')
all_nw.to_excel(r'output\定期更新\组合净值.xlsx')
