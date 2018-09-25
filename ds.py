#coding=utf-8

# 使用WIND数据库导入数据

import numpy as np
import pandas as pd
import re
from datetime import  datetime, timedelta
import pymongo
from constant import *
from mysql import connector
import matplotlib.pyplot as plt
import pyecharts as pec
import sympy



class deviation(object):


    def __init__(self, cmd_list, formula, needFex=0):

        # formula中的变量用var来代替
        self.cmd_list = cmd_list
        self.formula = formula

        # 根据正则来查看formula中有多少个变量
        ptn = re.compile('(?<=var)\d+')
        res = ptn.findall(self.formula)

        # 将公式进行合并多项式
        res = np.unique(res)
        for r in res:
            exec('var%s = sympy.Symbol("var%s")' % (r, r))
        self.formula = str(sympy.expand(self.formula))


        if len(self.cmd_list) != len(res):
            raise Exception(u'变量个数与公式中的变量个数不一致')

        self.conn = pymongo.MongoClient(host='localhost', port=27017)
        self.db = self.conn['FuturesDailyWind']

        if needFex:
            self.exchange = self.getForeignCurrency()
        else:
            self.exchange = None

        pd.set_option('display.max_columns', 30)


    def getForeignCurrency(self):

        db = self.conn['EDBWind']
        collection = db['FX']

        queryArgs = {'edb_name': '即期汇率:美元兑人民币'}
        projectionField = ['date', 'CLOSE']
        searchRes = collection.find(queryArgs, projectionField).sort('date', pymongo.ASCENDING)

        dt = []
        cls = []
        for r in searchRes:
            dt.append(r['date'])
            cls.append(r['CLOSE'])

        df = pd.DataFrame({'汇率': cls}, index=dt)
        return df

    @staticmethod
    def date_list():
        """用于生成日期的list，个数为366个，考虑了闰年的情况"""
        d_list = []
        dt = datetime(2016, 1, 1)
        while dt.year == 2016:
            d_list.append(dt.strftime('%m-%d'))
            dt = dt + timedelta(1)
        return d_list

    def seasonal_generator(self, df):
        # 将时间序列转成季节性矩阵
        price_diff = df[['price_diff']].copy()
        price_diff.dropna(inplace=True)
        df_index = price_diff.index
        price_diff['short_date'] = [d.strftime('%m-%d') for d in df_index]
        price_diff['year'] = [d.year for d in df_index]

        year_list = np.array([d.year for d in df_index])
        year_set = np.unique(year_list)

        dt_list = self.date_list()

        df_total = pd.DataFrame()
        for y in year_set:
            con = year_list == y
            df_year = price_diff.iloc[con]
            con = np.in1d(dt_list, df_year['short_date'].values)
            temp = np.ones(len(dt_list)) * np.nan
            temp[con] = df_year['price_diff'].values
            if df_total.empty:
                df_total = pd.DataFrame({y: temp}, index=dt_list)
            else:
                df_total = df_total.join(pd.DataFrame({y: temp}, index=dt_list), how='outer')

        df_total.fillna(method='ffill', inplace=True)

        return df_total

    def spotData(self, file_nm, spot_list):

        # spot_list的顺序与公式中变量的顺序是一致的
        # spotData里还有些问题，如果是用外盘数据的话

        if not file_nm.endswith('.csv'):
            file_nm += '.csv'
        df_csv = pd.read_csv(file_nm, index_col=0, parse_dates=True)

        spot_csv_list = [c for c in spot_list if c != '汇率']

        df_spot = df_csv[spot_csv_list].copy()

        n = len(spot_list)

        formula = self.formula
        for i in range(n):
            if spot_list[i] == '汇率':
                exec ('fex = self.getForeignCurrency()')
                formula = formula.replace('var%d' % (i + 1), 'fex["汇率"]')
            else:
                formula = formula.replace('var%d' % (i + 1), 'df_spot["%s"]' % spot_list[i])

        df_spot['price_diff'] = eval(formula)

        formulaCap = formula.replace('-', '+')
        formula_split = formulaCap.split('+')

        for part in formula_split:
            if 'df_spot' not in part:
                formula_split.remove(part)
        formulaCap = '+'.join(formula_split)
        df_spot['capital'] = eval(formulaCap)

        return df_spot

    # def futuresData(self):
    #     df = pd.DataFrame()
    #     com_list = []
    #     for i in range(len(self.cmd_list)):
            # 得到contract
            # exec('contract%d = main_contract["%s"]' % (i, self.cmd_list[i]))
            # exec('df%d = pd.DataFrame()' % i)
            # for s in eval('contract%d' % i):
            #     dftemp = self.same_month_combine(cmd=self.cmd_list[i], month=s)
            #     if eval('df%d' % i).empty:
            #         exec('df%d = dftemp.copy()' % i)
            #     else:
            #         exec('df%d = df%d.join(dftemp, how="outer")' % (i, i))

    def futuresMain(self):

        df_futures = pd.DataFrame()
        formula = self.formula
        for i in range(len(self.cmd_list)):
            if self.cmd_list[i] == '汇率':
                exec('contract%d = self.getForeignCurrency()' % i)
                formula = formula.replace('var%d' % (i + 1), 'contract%d["汇率"]' % i)
            else:
                exec('contract%d = self.get_price(collection="%s", wind_code="%s")' % (i, self.cmd_list[i] + '_Daily', self.cmd_list[i]))
                formula = formula.replace('var%d' % (i + 1), 'contract%d["close"]' % i)
                df_temp = eval('contract%d' % i).copy()
                df_temp.rename({'close': self.cmd_list[i]}, axis='columns', inplace=True)
                if df_futures.empty:
                    df_futures = df_temp.copy()
                else:
                    df_futures = df_futures.join(df_temp, how='outer')
        df_futures['price_diff'] = eval(formula)

        formulaCap = formula.replace('-', '+')
        formula_split = formulaCap.split('+')
        for part in formula_split:
            if 'contract' not in part:
                formula_split.remove(part)
        formulaCap = '+'.join(formula_split)
        df_futures['capital'] = eval(formulaCap)

        return df_futures

    def season_devi(self, df, rtn_len):

        # 传入的参数df是dataframe
        price_diff = df[['price_diff', 'capital']].copy()
        price_diff.dropna(inplace=True)

        df_index = price_diff.index
        price_diff['short_date'] = [d.strftime('%m-%d') for d in df_index]
        price_diff['year'] = [d.year for d in df_index]
        year_list = np.array([d.year for d in df_index])
        year_set = np.unique(year_list)

        df_total = self.seasonal_generator(df)

        df_mean = df_total.rolling(window=3, min_periods=3, axis=1).mean()
        df_mean = df_mean.shift(periods=1, axis=1)

        df_season = pd.DataFrame()

        for y in year_set:

            df_temp = df_mean[[y]].copy()

            if y % 4 != 0 or (y % 4 == 0 and y % 100 != 0):
                df_temp.drop(index='02-29', axis=0, inplace=True)

            df_temp['dt'] = [datetime.strptime(str(y) + '-' + d, '%Y-%m-%d') for d in df_temp.index]

            if df_season.empty:
                df_season = pd.DataFrame({'season_mean': df_temp[y].values}, index=df_temp['dt'].values)
            else:
                df_season = df_season.append(pd.DataFrame({'season_mean': df_temp[y].values},
                                                          index=df_temp['dt'].values))

        df_season = df_season.rolling(window=rtn_len, min_periods=rtn_len-10).mean()
        df_season = df_season.shift(periods=-rtn_len, axis=0)

        price_diff = price_diff.join(df_season, how='left')
        price_diff['profit'] = price_diff['season_mean'] - price_diff['price_diff']

        # df_col = df.columns.tolist()
        # df_col.remove('price_diff')
        # price_diff['avg_price'] = df[df_col].mean(axis=1)

        price_diff['profit_rate'] = price_diff['profit'] / price_diff['capital']

        price_diff['profit_rate_mean'] = price_diff[['profit_rate']].rolling(window=rtn_len, min_periods=rtn_len-10).mean()
        price_diff['profit_rate_std'] = price_diff[['profit_rate']].rolling(window=rtn_len, min_periods=rtn_len-10).std()

        price_diff['season_deviation_%ddays' % rtn_len] = (price_diff['profit_rate'] - price_diff['profit_rate_mean']) \
                                                        / price_diff['profit_rate_std']
        # print price_diff[['price_diff', 'season_deviation_%ddays' % rtn_len]]
        price_diff.to_clipboard()
        return price_diff[['price_diff', 'season_deviation_%ddays' % rtn_len]]

    def trend_devi(self, df, rtn_len):
        price_diff = df[['price_diff']].copy()
        price_diff_mean = price_diff.rolling(window=rtn_len, min_periods=rtn_len-10).mean()
        price_diff = price_diff.join(price_diff_mean, how='left', rsuffix='_mean')
        price_diff['price_diff_diff'] = price_diff['price_diff'] - price_diff['price_diff_mean']

        price_diff_std = price_diff[['price_diff']].rolling(window=rtn_len, min_periods=rtn_len-10).std()
        price_diff = price_diff.join(price_diff_std, how='left', rsuffix='_std')
        # print price_diff
        price_diff['trend_deviation_%ddays' % rtn_len] = - price_diff['price_diff_diff'] / price_diff['price_diff_std']

        return price_diff[['price_diff', 'trend_deviation_%ddays' % rtn_len]]

    def same_month_combine(self, cmd, month):
        # 相同月份的价格进行拼接，不过对于一些有两年合约的品种会有问题。
        # cmd是品种，如'TA.CZC'；month是月份，如'01'

        col = self.db[cmd + '_Daily']
        res = col.distinct('wind_code', filter={'wind_code':{'$regex': '.*%s(?=\.)' % month}})
        res.sort(reverse=False)
        df = pd.DataFrame()
        for r in res:
            if df.empty:
                df =  self.get_price(cmd + '_Daily', r)
            else:
                df = df.append(self.get_price(cmd + '_Daily', r))
        df.rename(lambda x: x + '_' + month + '_' + cmd, axis='columns', inplace=True)
        return df

    def get_price(self, collection, wind_code):

        """collection是集合的名字, wind_code是WIND代码，该函数用于调取期货合约的收盘价"""

        # if 'db' not in vars(self) or self.db.name != 'FuturesDailyWind':
        #     self.db = self.conn['FuturesDailyWind']

        col = self.db[collection]
        queryArgs = {'wind_code': wind_code}
        projectionField = ['date', 'CLOSE']
        res = col.find(queryArgs, projectionField).sort('date', pymongo.ASCENDING)

        cls = []
        dt = []

        for r in res:
            dt.append(r['date'])
            cls.append(r['CLOSE'])

        df = pd.DataFrame({'close': cls}, index=dt)
        return df

    def get_future_end_date(self, wind_code):

        # if 'db' not in vars(self) or self.db.name != 'FuturesDailyWind':
        #     self.db = self.conn['FuturesDailyWind']
        col = self.db['FuturesInfo']
        queryArgs = {'wind_code': wind_code}
        projectionField = ['last_trade_date']
        res = col.find(queryArgs, projectionField)
        for r in res:
            dt = r['last_trade_date']
        return dt

    def plot_devi_price(self, df, rtn_len, corr_len, mode='trend'):

        plt.figure()
        plt.subplot(211)

        if mode == 'trend':
            df_plot = self.trend_devi(df, rtn_len)
            df_plot['trend_deviation_%ddays' % rtn_len].plot(color='k')
            df_corr = df_plot['trend_deviation_%ddays' % rtn_len].rolling(window=corr_len, min_periods=corr_len-10).\
                corr(df_plot['price_diff'], pairwise=False)
            df_corr = df_corr.to_frame('RollingCorr_%ddays' % corr_len)
        elif mode == 'season':
            df_plot = self.season_devi(df, rtn_len)
            df_plot['season_deviation_%ddays' % rtn_len].plot(color='k')
            df_corr = df_plot['season_deviation_%ddays' % rtn_len].rolling(window=corr_len, min_periods=corr_len-10).\
                corr(df_plot['price_diff'], pairwise=False)
            df_corr = df_corr.to_frame('RollingCorr_%ddays' % corr_len)
        else:
            raise Exception(u'错误的参数输入')

        plt.legend()
        ax = plt.gca()
        ax2 = plt.twinx()
        df_plot['price_diff'].plot(color='r')
        plt.grid()
        plt.legend()

        plt.subplot(212, sharex=ax)

        df_corr['RollingCorr_%ddays' % corr_len].plot(color='r')
        plt.legend()
        plt.grid()
        print 'Pearson:', df_plot.corr(method='pearson')
        print 'Spearman:', df_plot.corr(method='spearman')
        # plt.show()

    def echart_devi_price(self, df, rtn_len, corr_len, mode='trend'):
        line1 = pec.Line(str(self.cmd_list))
        line2 = pec.Line()
        line3 = pec.Line(str(self.cmd_list))
        if mode == 'trend':
            df_res = self.trend_devi(df, rtn_len)
            df_res.dropna(how='all', inplace=True)
            ds_ = df_res['trend_deviation_%ddays' % rtn_len].tolist()
            df_corr = df_res['trend_deviation_%ddays' % rtn_len].rolling(window=corr_len, min_periods=corr_len-10).\
                corr(df_res['price_diff'], pairwise=False)
            df_corr = df_corr.to_frame('RollingCorr_%ddays' % corr_len)
        elif mode == 'season':
            df_res = self.season_devi(df, rtn_len)
            df_res.dropna(how='all', inplace=True)
            ds_ = df_res['season_deviation_%ddays' % rtn_len].tolist()
            df_corr = df_res['season_deviation_%ddays' % rtn_len].rolling(window=corr_len, min_periods=corr_len-10).\
                corr(df_res['price_diff'], pairwise=False)
            df_corr = df_corr.to_frame('RollingCorr_%ddays' % corr_len)
        else:
            raise Exception(u'错误的参数输入')

        dt_ = df_res.index.tolist()
        price_ = df_res['price_diff'].tolist()
        corr_ = df_corr['RollingCorr_%ddays' % corr_len].tolist()

        line1.add('%s_deviation_%ddays' % (mode, rtn_len), dt_, ds_, is_datazoom_show=True, datazoom_type='both',
                  tooltip_trigger='axis')

        line2.add('price_diff', dt_, price_)

        overlap = pec.Overlap()
        overlap.add(line1)
        overlap.add(line2, is_add_yaxis=True, yaxis_index=1)

        line3.add('RollingCorr_%ddays' % corr_len, dt_, corr_, is_datazoom_show=True, datazoom_type='both',
                  tooltip_trigger='axis')

        return overlap, line3

    def basisData(self, file_nm, spot_list):
        spot = self.spotData(file_nm, spot_list)[['price_diff']]
        future = self.futuresMain()[['price_diff']]
        basis = spot - future
        return basis


def get_price(collection, wind_code, mongo_host='localhost', mongo_port=27017, field='CLOSE'):

    """collection是集合的名字, wind_code是WIND代码，该函数用于调取期货合约的数据"""

    # if 'db' not in vars(self) or self.db.name != 'FuturesDailyWind':
    #     self.db = self.conn['FuturesDailyWind']

    conn = pymongo.MongoClient(host=mongo_host, port=mongo_port)
    db = conn['FuturesDailyWind']

    col = db[collection]
    queryArgs = {'wind_code': wind_code}
    projectionField = ['date', field]
    res = col.find(queryArgs, projectionField).sort('date', pymongo.ASCENDING)

    cls = []
    dt = []

    for r in res:
        dt.append(r['date'])
        cls.append(r['CLOSE'])

    df = pd.DataFrame({field: cls}, index=dt)
    return df


class CorrelCalc(object):

    def __init__(self, cmd_list, mongo_host='localhost', mongo_port=27017):

        self.cmd_list = cmd_list
        self.conn = pymongo.MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.conn['FuturesDailyWind']

        if len(self.cmd_list) != 2:
            raise Exception(u'请输入两个待比较的名称')

        self.ts1 = get_price(collection=self.cmd_list[0]+'_Daily', wind_code=self.cmd_list[0])
        self.ts1.rename({'CLOSE': self.cmd_list[0]}, axis='columns', inplace=True)
        self.ts2 = get_price(collection=self.cmd_list[1] + '_Daily', wind_code=self.cmd_list[1])
        self.ts2.rename({'CLOSE': self.cmd_list[1]}, axis='columns', inplace=True)
        self.total_ts = self.ts1.join(self.ts2, how='outer')

        self.rtn_ts = self.total_ts.pct_change()


    def correl_total(self, method='pearson'):
        print self.total_ts.corr(method=method)
        print self.rtn_ts.corr(method=method)




if __name__ == '__main__':

    # a = CorrelCalc(cmd_list=['L.DCE', 'PP.DCE'])
    # a.correl_total()



    formula = "var1 - var2 + 3 * var2 - 3 * var2 * var3"
    # formula2 =

    a = deviation(cmd_list=['L.DCE', 'PP.DCE', '汇率'], formula=formula, needFex=0)
    # a.getForeignCurrency()
    # a.futuresMain()
    a.season_devi(a.spotData('raw_data', spot_list=['LL神华', 'PP华东', '汇率']), 60)
    # print a.season_devi(df=a.futuresMain(), rtn_len=60)
    # print a.futuresMain()
    # a.plot_devi_price(df=a.futuresMain(), rtn_len=60, corr_len=60, mode='season')
    # plt.show()
    # a.echart_devi_price(a.futuresMain(), rtn_len=60, corr_len=60)
    # a.plot_devi_price(a.futuresMain(), 60, 60)

    # a = deviation(cmd_list=['BU.SHF', 'TA.CZC'], formula=formula, needFex=0)
    # a.spotData('raw_data', spot_list=['LL神华', 'PP华东'])
    # print a.get_price_wind(collection='TA.CZC_Daily', wind_code='TA901.CZC')
    # print a.same_month_combine('TA.CZC', '01')
    # a.seasonal_generator(a.futuresMain())
    # print a.seasonal_generator(a.spotData('raw_data', spot_list=['LL神华', 'PP华东']))
    # print a.spotData()
    # a.trend_devi(a.futuresMain(), rtn_len=90)['trend_deviation_90days'].plot()
    # # a.season_devi(a.spotData('raw_data', spot_list=['LL神华', 'PP华东']), rtn_len=60).plot(color='k')
    # ax = plt.gca()
    # ax.grid()
    # ax.legend()
    #
    # ax2 = plt.twinx()
    # a.futuresMain()['price_diff'].plot(color='r')
    # ax2.legend()
    #
    # a.trend_devi(a.futuresMain(), rtn_len=21)
    #
    # plt.show()

    # print a.same_month_combine('jq',cmd='L', month='01')

    # df1 = a.futuresMain()
    # df2 = a.spotData('raw_data', spot_list=['LL神华', 'PP华东'])


    # a1 = a.trend_devi(df1, rtn_len=90)[['trend_deviation_90days']]
    # b1 = a.trend_devi(df2, rtn_len=90)[['trend_deviation_90days']]
    # c1 = a1 / b1
    # print c1.dropna()
    # print a1.index
    # print b1.index
    # print a.trend_devi(df1, rtn_len=90)[['trend_deviation_90days']]
    # print a.trend_devi(df2, rtn_len=90)[['trend_deviation_90days']]
    # a.plot_devi_price(df1, rtn_len=90, corr_len=90, mode='trend')

    # a1 = a.basisData('raw_data', spot_list=['LL神华', 'PP华东'])
    # a.plot_devi_price(a1, rtn_len=90, corr_len=90, mode='season')
    # b1 = a.futuresMain()
    # df1 = a.season_devi(b1, rtn_len=90)
    #
    # df1.to_csv('bu-ta.csv')
    # a.plot_devi_price(b1, rtn_len=90, corr_len=90, mode='season')
    # c1 = a.spotData('raw_data', spot_list=['LL神华', 'PP华东'])
    # a.plot_devi_price(c1, rtn_len=90, corr_len=90, mode='season')
    plt.show()