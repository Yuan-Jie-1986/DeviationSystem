#coding=utf-8

# Author: YuanJie

import os
import inspect
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from constant import *
import pymongo
from mysql import connector


class PriceSeizing(object):

    def __init__(self, host='localhost', port=27017):
        self.conn = pymongo.MongoClient(host=host, port=port)


    def useMongoDB(self, db):
        self.db = self.conn[db]

    def useMongoCollections(self, collection):
        if self.db:
            self.collection = self.db[collection]
        else:
            raise Exception(u'请先选择数据库')

    # 根据合约提取数据
    def getPriceAsContract(self, collections, jq_code, field=['close']):
        self.useMongoDB('FuturesDaily')
        self.useMongoCollections(collections)
        queryArgs = {'jq_code': jq_code}
        projectionFields = ['date']
        projectionFields.extend(field)

        res = self.collection.find(queryArgs, projectionFields).sort('date', pymongo.ASCENDING)

        for f in field:
            exec (f + ' = []')

        dt = []

        for r in res:
            dt.append(r['date'])
            for f in field:
                eval(f).append(r[f])

        f_dict = {}
        for f in field:
            f_dict[f] = eval(f)
        df = pd.DataFrame(f_dict, index=dt)
        return df


class Deviation(object):

    def __init__(self, file_nm, cmd_list, formula, needEx=0):
        # formula中的变量用var*代替
        self.cmd_list = cmd_list
        self.formula = formula

        # 根据正则来查看formula中有多少个变量
        ptn = re.compile('(?<=var)\d+')
        res = ptn.findall(self.formula)

        if len(self.cmd_list) != len(res):
            raise Exception(u'变量个数与公式中的变量个数不一致')

        if not file_nm.endswith('.csv'):
            file_nm += '.csv'

        self.cmd_list_csv = [c for c in self.cmd_list if c != '汇率']

        df_csv = pd.read_csv(file_nm, index_col=0, parse_dates=True)
        self.df_lp = df_csv[self.cmd_list_csv].copy()

        if needEx:
            self.exchange = self.getForeignCurrency()
            self.df_lp = self.df_lp.join(self.exchange, how='left')
            self.df_lp.fillna(method='ffill', inplace=True)

        n = len(self.cmd_list)

        formula1 = self.formula

        for i in range(n):
            formula1 = formula1.replace('var' + str(i + 1), 'self.df_lp["' + self.cmd_list[i] + '"]')
        self.df_lp['price_diff'] = eval(formula1)

        self.conn = pymongo.MongoClient(host='localhost', port=27017)
        self.db = self.conn['FuturesDailyWind']

        pd.set_option('display.max_columns', 30)

    def getForeignCurrency(self):
        # 从mysql数据库抓取
        conn = connector.connect(user='root', password='cbnb888')
        conn.database = 'macroeco'
        cursor = conn.cursor()

        sql = 'select date, name, value from currency where name="中间价:美元兑人民币" order by date'
        cursor.execute(sql)
        res = cursor.fetchall()

        dt = [r[0] for r in res]
        val = [r[2] for r in res]

        df = pd.DataFrame({'汇率': val}, index=dt)

        return df

    def price_average(self):

        self.df_lp['year'] = [d.year for d in self.df_lp.index]
        res = self.df_lp.groupby('year').mean()
        return res

    def seasonal_generator(self):
        # 将时间序列转成季节性矩阵
        price_diff = self.df_lp[['price_diff']].copy()
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

    def seasonal_devi(self):
        price_diff = self.df_lp[['price_diff']].copy()
        df_index = price_diff.index
        price_diff['short_date'] = [d.strftime('%m-%d') for d in df_index]
        price_diff['year'] = [d.year for d in df_index]
        year_list = np.array([d.year for d in df_index])
        year_set = np.unique(year_list)

        df_total = self.seasonal_generator()

        # df_total.to_clipboard(excel=True)

        df_mean = df_total.rolling(window=3, min_periods=3, axis=1).mean()
        df_mean = df_mean.shift(periods=1, axis=1)

        # df_mean.to_clipboard()
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

        # print df_season
        # con = np.in1d(df_season.index, df_index)
        # df_season.drop(df_season.index[~con], axis=0, inplace=True)

        # print df_season.loc[datetime(2018,3,30)]
        df_season = df_season.rolling(window=90, min_periods=80).mean()
        df_season = df_season.shift(periods=-90, axis=0)
        df_season.dropna(axis=0, inplace=True)
        price_diff = price_diff.join(df_season, how='left')

        price_diff['profit'] = price_diff['season_mean'] - price_diff['price_diff']
        price_avg = self.price_average()
        price_avg.drop(columns='price_diff', inplace=True)
        price_avg = price_avg.shift(periods=1, axis=0)
        price_diff = price_diff.join(price_avg, on='year', how='left')

        price_diff['avg_price'] = price_diff[self.cmd_list].mean(axis=1)


        price_diff['profit_rate'] = price_diff['profit'] / price_diff['avg_price']
        # price_diff['profit_rate'].plot()
        # plt.show()
        profit_mean = price_diff[['profit_rate', 'year']].groupby('year').mean()
        profit_mean = profit_mean.shift(periods=1, axis=0)
        profit_std = price_diff[['profit_rate', 'year']].groupby('year').std()
        profit_std = profit_std.shift(periods=1, axis=0)

        price_diff = price_diff.join(profit_mean, on='year', how='left', rsuffix='_mean')
        price_diff = price_diff.join(profit_std, on='year', how='left', rsuffix='_std')

        price_diff['season_deviation'] = (price_diff['profit_rate'] - price_diff['profit_rate_mean']) \
                                         / price_diff['profit_rate_std']

        # price_diff[['season_deviation']].iloc[-400:].plot()
        # plt.grid()
        # plt.show()

        return price_diff[['season_deviation']]

    def seasonal_devi2(self, para):
        """para参数是用于进行均值和标准差计算的滚动周期"""
        price_diff = self.df_lp[['price_diff']].copy()
        df_index = price_diff.index
        price_diff['short_date'] = [d.strftime('%m-%d') for d in df_index]
        price_diff['year'] = [d.year for d in df_index]
        year_list = np.array([d.year for d in df_index])
        year_set = np.unique(year_list)

        df_total = self.seasonal_generator()

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

        df_season = df_season.rolling(window=90, min_periods=80).mean()
        df_season = df_season.shift(periods=-90, axis=0)
        df_season.dropna(axis=0, inplace=True)

        price_diff = price_diff.join(df_season, how='left')

        price_diff['profit'] = price_diff['season_mean'] - price_diff['price_diff']
        price_diff['avg_price'] = self.df_lp[self.cmd_list_csv].mean(axis=1)

        price_diff['profit_rate'] = price_diff['profit'] / price_diff['avg_price']

        price_diff['profit_rate_mean'] = price_diff[['profit_rate']].rolling(window=para, min_periods=para-9).mean()
        price_diff['profit_rate_std'] = price_diff[['profit_rate']].rolling(window=para, min_periods=para-9).std()

        price_diff['season_deviation_%ddays' % para] = (price_diff['profit_rate'] - price_diff['profit_rate_mean']) \
                                                       / price_diff['profit_rate_std']

        return price_diff[['season_deviation_%ddays' % para]]

    def trend_devi(self):
        price_diff = self.df_lp[['price_diff']].copy()
        price_diff_mean = price_diff.rolling(window=21, min_periods=11).mean()
        price_diff_mean = price_diff_mean.shift(periods=1, axis=0)
        price_diff = price_diff.join(price_diff_mean, how='left', rsuffix='_mean')
        price_diff['price_diff_diff'] = price_diff['price_diff'] - price_diff['price_diff_mean']

        price_diff_std = price_diff[['price_diff_diff']].rolling(window=252, min_periods=11).std()
        price_diff_std = price_diff_std.shift(periods=1, axis=0)
        price_diff = price_diff.join(price_diff_std, how='left', rsuffix='_std')
        price_diff['trend_deviation'] = - price_diff['price_diff_diff'] / price_diff['price_diff_diff_std']

        return price_diff[['trend_deviation']]

    def __same_month_combine(self, cmd, month):
        # 将相同月份的合约进行拼接，但是对于一些有两年合约的品种是有问题的。
        # cmd比如'TA', month比如'09'
        col = self.db[cmd + '_Daily']
        res = col.distinct('wind_code', filter={'wind_code':{'$regex': '.*%s(?=\.)' % month}})
        res.sort(reverse=False)
        df = pd.DataFrame()
        for r in res:
            if df.empty:
                df = self.get_price(cmd + '_Daily', r)
            else:
                df = df.append(self.get_price(cmd + '_Daily', r))
        df.rename(lambda x: x+ '_' + month + '_' + cmd, axis='columns', inplace=True)
        return df

    def __find_price_diff_counterparty(self, *args):
        # 根据合约的list寻找配对的价差

        list_len = [len(a) for a in args]
        if len(np.unique(list_len)) == 1:
            return zip(*args)
        elif 1 in list_len:
            list_len.remove(1)
            n = np.unique(list_len)
            if len(n) == 1:
                n = n[0]
                new_list = []
                for a in args:
                    print a
                    if len(a) == 1:
                        a = a * n
                        new_list.append(a)
                    else:
                        new_list.append(a)
                return zip(*new_list)

        else:
            return None

    def func_year(self, x):
        return len(set(x)) == 1

    def adjust_devi(self, fut_list):

        if len(fut_list) != len(self.cmd_list_csv):
            raise Exception(u'期货个数与现货个数不相同')

        fut_list_total = np.array(self.cmd_list).copy()
        fut_list_total[np.in1d(self.cmd_list, self.cmd_list_csv)] = fut_list

        df = pd.DataFrame()

        com_list = []

        for i in range(len(fut_list)):

            # 得到contract*

            exec('contract%d = main_contract["%s"]' % (i, fut_list[i]))
            exec('df%d = pd.DataFrame()' % i)

            if len(eval('contract%d'%i)) != 1:

                for s in eval('contract%d' % i):
                    dftemp = self.__same_month_combine(cmd=fut_list[i], month=s)
                    if eval('df%d' % i).empty:
                        exec('df%d = dftemp.copy()' % i)
                    else:
                        exec('df%d = df%d.join(dftemp, how="outer")' % (i, i))
                    # print eval('df%d' % i)
            elif len(eval('contract%d' % i)) == 1:
                conval = eval('contract%d' % i)[0]
                col = self.db['%s_Daily' % conval]

                queryArgs = {'wind_code': conval}
                projectionField = ['date', 'CLOSE']
                res = col.find(queryArgs, projectionField).sort('date', pymongo.ASCENDING)
                dt = []
                cls = []
                for r in res:
                    dt.append(r['date'])
                    cls.append(r['CLOSE'])
                exec('df%d = pd.DataFrame({"CLOSE_" + conval: cls}, index=dt)' % i)

            if df.empty:
                df = eval('df%d' % i).copy()
            else:
                df = df.join(eval('df%d' % i), how='inner')

            com_list.append(eval('contract%d' % i))
            # print com_list

        pairs = self.__find_price_diff_counterparty(*com_list)

        # df_remain是每个差价合约的剩余天数。
        # 针对月份如果不同的合约，会根据当期的年份不同而赋成nan
        df_remain = pd.DataFrame()

        for p in pairs:
            key_name = str(p)
            year_col = ''
            ptn_p = re.compile('\d+')
            calc_name = self.formula

            p_new = np.array(fut_list_total).copy()
            p_new[np.in1d(fut_list_total, fut_list)] = p

            for i in range(len(p_new)):
                if ptn_p.search(p_new[i]):
                    p_temp = '%s_%s' % (ptn_p.search(p_new[i]).group(), fut_list_total[i])
                    if year_col:
                        year_col += ','
                    year_col += 'end_year_%s' % (p_temp)
                else:
                    p_temp = p_new[i]
                if p_temp == '汇率':
                    calc_name = calc_name.replace('var%d' % (i + 1), 'self.df_lp["汇率"]')
                else:
                    calc_name = calc_name.replace('var%d' % (i + 1), 'df["CLOSE_%s"]' % p_temp)

            year_col = year_col.split(',')
            df_end_year = df[year_col]
            same_year = df_end_year.apply(self.func_year, axis=1)

            df[key_name] = eval(calc_name)
            col_p = []
            for i in range(len(p)):
                if ptn_p.search(p[i]):
                    col_p.append('remain_days_%s_%s' % (p[i], fut_list[i]))

            if df_remain.empty:
                df_remain = pd.DataFrame({key_name + ' RD': df[col_p].min(axis=1, skipna=False)})

                df_remain[~same_year] = np.nan
            else:
                df_remain_temp = pd.DataFrame({key_name + ' RD': df[col_p].min(axis=1, skipna=False)})
                df_remain_temp[~same_year] = np.nan
                df_remain = df_remain.join(df_remain_temp)


        df_rank =  df_remain.rank(axis=1)
        df_remain_day = pd.DataFrame({'remain': [d.days for d in df_remain.min(axis=1, skipna=False)]},
                                     index=df_remain.index)
        front1 = np.ones(len(df_rank)) * np.nan
        front2 = np.ones(len(df_rank)) * np.nan
        front3 = np.ones(len(df_rank)) * np.nan

        p1 = re.compile('.+(?=\sRD)')
        for i in range(len(df_rank)):
            df_temp = df_rank.iloc[i]

            temp1 = df_temp[df_temp==1].index
            if temp1.empty:
                front1[i] = np.nan
            else:
                find_res = p1.search(temp1[0]).group()
                front1[i] = df[find_res].iloc[i]

            temp2 = df_temp[df_temp==2].index

            if temp2.empty:
                front2[i] = np.nan
            else:
                find_res = p1.search(temp2[0]).group()
                front2[i] = df[find_res].iloc[i]

            temp3 = df_temp[df_temp == 3].index
            if temp3.empty:
                front3[i] = np.nan
            else:
                find_res = p1.search(temp3[0]).group()
                front3[i] = df[find_res].iloc[i]

        df['front1'] = front1
        df['front2'] = front2
        df['front3'] = front3

        df = df.join(self.df_lp, how='left')
        df['front1-commodity'] = df['front1'] - df['price_diff']
        df['front2-commodity'] = df['front2'] - df['price_diff']
        df['front3-commodity'] = df['front3'] - df['price_diff']

        df_total = self.seasonal_generator()
        df_std = df_total.rolling(window=3, axis=1).std()
        df_std = df_std.shift(periods=1, axis=1)


        df['year'] = [d.year for d in df.index]

        df_std = pd.DataFrame({'season_std_mean': df_std.mean()})

        df = df.join(df_std, on='year')
        df = df.join(self.seasonal_devi2(para=90), how='left')
        df['front1_season_std'] = df['season_deviation_90days'] - df['front1-commodity'] / df['season_std_mean']
        df['front2_season_std'] = df['season_deviation_90days'] - df['front2-commodity'] / df['season_std_mean']
        df['front3_season_std'] = df['season_deviation_90days'] - df['front3-commodity'] / df['season_std_mean']

        df = df.join(df_remain_day, how='left')
        df['total_wgt'] = 1. / np.sqrt(df['remain'] + 10.) + 1. / np.sqrt(df['remain'] + 120.) + \
                          1. / np.sqrt(df['remain'] + 240.)
        df['wgted_season_deviation'] =  (1. / np.sqrt(df['remain'] + 10.) / df['total_wgt'] * df['front1_season_std']) + \
                                        (1. / np.sqrt(df['remain'] + 120.) / df['total_wgt'] * df['front2_season_std']) + \
                                        (1. / np.sqrt(df['remain'] + 240.) / df['total_wgt'] * df['front3_season_std'])

        price_diff = self.df_lp[['price_diff']].copy()
        price_diff_mean = price_diff.rolling(window=21, min_periods=11).mean()
        price_diff_mean = price_diff_mean.shift(periods=1, axis=0)
        price_diff = price_diff.join(price_diff_mean, how='left', rsuffix='_mean')
        price_diff['price_diff_diff'] = price_diff['price_diff'] - price_diff['price_diff_mean']

        price_diff_std = price_diff[['price_diff_diff']].rolling(window=252, min_periods=11).std()
        price_diff_std = price_diff_std.shift(periods=1, axis=0)
        price_diff = price_diff.join(price_diff_std, how='left', rsuffix='_std')
        price_diff['trend_deviation'] = - price_diff['price_diff_diff'] / price_diff['price_diff_diff_std']

        df = df.join(price_diff[['price_diff_diff_std', 'trend_deviation']], how='left')

        df['front1_trend_std'] = df['trend_deviation'] - df['front1-commodity'] / df['price_diff_diff_std']
        df['front2_trend_std'] = df['trend_deviation'] - df['front2-commodity'] / df['price_diff_diff_std']
        df['front3_trend_std'] = df['trend_deviation'] - df['front3-commodity'] / df['price_diff_diff_std']

        df['wgted_trend_deviation'] = (1. / np.sqrt(df['remain'] + 10.) / df['total_wgt'] * df['front1_trend_std']) + \
                                      (1. / np.sqrt(df['remain'] + 120.) / df['total_wgt'] * df['front2_trend_std']) + \
                                      (1. / np.sqrt(df['remain'] + 240.) / df['total_wgt'] * df['front3_trend_std'])

        # df[['season_deviation', 'wgted_season_deviation']].iloc[-400:].plot()
        # df[['trend_deviation', 'wgted_trend_deviation']].iloc[-400:].plot()
        # plt.show()

        return df[['wgted_season_deviation', 'wgted_trend_deviation']]

    def plotWgtedDevi(self, fut_list):
        df1 = self.adjust_devi(fut_list)
        df2 = self.seasonal_devi2(para=90)
        df3 = self.trend_devi()

        df = df1.join(df2)
        df = df.join(df3)

        df[['season_deviation_90days', 'wgted_season_deviation']].plot()
        plt.grid()
        df[['trend_deviation', 'wgted_trend_deviation']].plot()

        plt.grid()
        plt.show()

    def get_price(self, collection, wind_code):

        col = self.db[collection]
        queryArgs = {'wind_code': wind_code}
        projectionField = ['date', 'CLOSE']
        res = col.find(queryArgs, projectionField).sort('date', pymongo.ASCENDING)

        cls = []
        dt = []

        for r in res:
            dt.append(r['date'])
            cls.append(r['CLOSE'])

        df = pd.DataFrame({'CLOSE': cls}, index=dt)

        df['end_date'] = self.get_future_end_date(wind_code=wind_code)
        df['remain_days'] = df['end_date'] - df.index

        end_year = []
        for i in range(len(df)):
            end_year.append(df.iloc[i]['end_date'].year)

        df['end_year'] = end_year

        return df

    def get_future_end_date(self, wind_code):

        col = self.db['FuturesInfo']
        queryArgs = {'wind_code': wind_code}
        projectionField = ['last_trade_date']
        res = col.find(queryArgs, projectionField)
        for r in res:
            dt = r['last_trade_date']
        return dt

    @staticmethod
    def date_list():
        """用于生成日期的list，个数为366个，考虑了闰年的情况"""
        d_list = []
        dt = datetime(2016, 1, 1)
        while dt.year == 2016:
            d_list.append(dt.strftime('%m-%d'))
            dt = dt + timedelta(1)
        return d_list


if __name__ == '__main__':
    # a = Deviation(file_nm='raw_data', cmd_list=['LL神华', 'PP华东'], para_list=[1, -1], needEx=1)
    formula = "var1 - (var2 * 7.5 + 90 + 400) * 1.02 * 1.16 * 0.656 * var3"
    a = Deviation(file_nm='ta_brent', cmd_list=['TA', 'BRENT', '汇率'], formula=formula, needEx=1)

    # df1 = a.seasonal_devi2(756)
    #
    # # df2 = df1.join(a.seasonal_devi2(60))
    # # df2.plot()
    # df2 = df1.join(a.df_lp['price_diff'], how='left')
    # df2.plot(secondary_y=['price_diff'])
    # plt.grid()
    #
    # plt.show()
    # a = Deviation(file_nm='ferrous_data', cmd_list=['PB粉', '焦炭', '螺纹钢'], para_list=[-1.7, -0.5, 1])
    # a.adjust_devi(fut_list=['L', 'PP'])
    # a.plotWgtedDevi(fut_list=['I', 'J', 'RB'])
    # a.plotWgtedDevi(fut_list=['L', 'PP'])
    # a.find_price_diff_counterparty(main_contract['TA'], main_contract['L'])
    # a.price_average()

    # Deviation(cmd1='LL神华', cmd2='PP华东').get_price('L_Daily', 'L1801.XDCE')
    # print a.seasonal_generator()
    # a.adjust_devi(['TA.CZC', 'B.IPE'])
    a.plotWgtedDevi(['TA.CZC', 'B.IPE'])
    # df = a.seasonal_devi2(para=90)
    # df.plot()
    # plt.show()
