import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')


# 加载数据
def load_data():
    train_data = pd.read_csv('../userdata/data_for_tree.csv')
    return train_data


# 对数据进行处理，减少空间的使用
def reduce_mem_usage(train_data):
    # 最开始所占空间
    start_mem = train_data.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in train_data.columns:
        col_type = train_data[col].dtype

        if col_type != object:
            c_min = train_data[col].min()
            c_max = train_data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    train_data[col] = train_data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    train_data[col] = train_data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    train_data[col] = train_data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    train_data[col] = train_data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    train_data[col] = train_data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    train_data[col] = train_data[col].astype(np.float32)
                else:
                    train_data[col] = train_data[col].astype(np.float64)
        else:
            train_data[col] = train_data[col].astype('category')

    end_mem = train_data.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return train_data


# 准备训练数据
def prepare_data(train_data):
    # 获取相关特征名
    feature_names = [x for x in train_data.columns if x not in ['price', 'brand', 'model']]

    # 修改数据
    train_data = train_data.dropna().replace('-', 0).reset_index(drop=True)
    train_data['notRepairedDamage'] = train_data['notRepairedDamage'].astype(np.float32)

    # 构建相关训练项
    train = train_data[feature_names + ['price']]
    train_x = train[feature_names]
    train_y = train['price']

    return train_data, train_x, train_y, feature_names


# 简单线性模型
def simple_linear_model(train_data, train_x, train_y, feature_names):
    model = LinearRegression(normalize=True)
    model = model.fit(train_x, train_y)
    # 查看所得模型的截距和权重
    intercept = str(model.intercept_)
    print("intercept：{}".format(intercept))
    weight = sorted(dict(zip(feature_names, model.coef_)).items(), key=lambda x: x[1], reverse=True)
    print(weight)

    # 可视化处理
    # 在0-len(train_y)之间，随机生成50个随机数，用做index索引
    subsample_index = np.random.randint(low=0, high=len(train_y), size=50)
    predict = model.predict(train_x.loc[subsample_index])

    # 对特征V_9进行可视化
    v_9(train_x, train_y, subsample_index, predict)
    # 经过可视化可以发现，分布的效果并不好

    # 在数据分析阶段，我们知道price的分布不符合正太分布，故对标签进行log(x+1)变换，并进行可视化
    train_y_ln = np.log(train_y+1)
    # 可视化
    price(train_y_ln)
    # 由可视化图可知，转换后的标签更满足正态分布

    # 重新对模型进行处理
    model = model.fit(train_x, train_y_ln)
    # 查看新得模型的截距和权重
    intercept = str(model.intercept_)
    print("intercept：{}".format(intercept))
    weight = sorted(dict(zip(feature_names, model.coef_)).items(), key=lambda x: x[1], reverse=True)
    print(weight)

    # 可视化处理
    # 在0-len(train_y_ln)之间，随机生成50个随机数，用做index索引
    subsample_index = np.random.randint(low=0, high=len(train_y_ln), size=50)
    predict = np.exp(model.predict(train_x.loc[subsample_index]))

    # 对特征V_9进行可视化
    v_9(train_x, train_y, subsample_index, predict)
    # 由新的可视化图可以看到，所得模型分布较好，且未出现异常值

    # 五折交叉验证
    half_cross(model, train_x, train_y, train_y_ln)

    # 模拟真实业务情况
    real_simulate(train_data, feature_names, model)
    # 结果显示与五折交叉验证差距不大

    return model


# 可视化V_9，画出散点图
def v_9(train_x, train_y, subsample_index, predict):
    plt.scatter(train_x['v_9'][subsample_index], train_y[subsample_index], color='black')
    plt.scatter(train_x['v_9'][subsample_index], predict, color='blue')
    plt.xlabel('v_9')
    plt.ylabel('price')
    plt.legend(['True Price', 'Predicted Price'], loc='upper right')
    plt.show()


# 可视化变换后的price
def price(train_y_ln):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.distplot(train_y_ln)
    plt.subplot(1, 2, 2)
    sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])
    plt.show()


def log_transfer(func):
    def wrapper(y, yhat):
        result = func(np.log(y), np.nan_to_num(np.log(yhat)))
        return result
    return wrapper


# 五折交叉
def half_cross(model, train_x, train_y, train_y_ln):
    # 对未处理的标签进行五折交叉验证
    scores = cross_val_score(model, X=train_x, y=train_y, verbose=1, cv=5,
                             scoring=make_scorer(log_transfer(mean_absolute_error)))
    print(scores)
    print('AVG', np.mean(scores))

    # 对处理过的标签进行交叉验证
    scores = cross_val_score(model, X=train_x, y=train_y_ln, verbose=1, cv=5, scoring=make_scorer(mean_absolute_error))
    print(scores)
    print('AVG', np.mean(scores))

    scores = pd.DataFrame(scores.reshape(1, -1))
    scores.columns = ['cv' + str(x) for x in range(1, 6)]
    scores.index = ['MAE']
    print(scores)


# 模拟真实情况
def real_simulate(train_data, feature_names, model):
    train_data = train_data.reset_index(drop=True)
    split_point = len(train_data) // 5 * 4
    train = train_data.loc[:split_point].dropna()
    val = train_data.loc[split_point:].dropna()

    train_x = train[feature_names]
    train_y_ln = np.log(train['price'] + 1)
    val_x = val[feature_names]
    val_y_ln = np.log(val['price'] + 1)

    model = model.fit(train_x, train_y_ln)
    print(mean_absolute_error(val_y_ln, model.predict(val_x)))


def predict_y(test_data, model):
    # 构造数据
    predict = pd.DataFrame()
    predict['SaleID'] = test_data['SaleID']
    test_data['train'] = 1
    test_data['price'] = 1
    # test_data['city'] = 1

    # 使用时间
    car_use_time = pd.to_datetime(test_data['creatDate'], format='%Y%m%d', errors='coerce')
    car_reg_time = pd.to_datetime(test_data['regDate'], format='%Y%m%d', errors='coerce')
    test_data['used_time'] = (car_use_time - car_reg_time).dt.days

    # 从邮编中提取城市信息
    # test_data['city'] = (test_data['regionCode'].apply(lambda x: str(x)[:-3]))

    # 计算某个品牌的销售量
    test_data_gb = test_data.groupby("brand")  # 根据brand对数组进行分类
    all_info = {}
    for kind, kind_data in test_data_gb:
        info = {}
        kind_data = kind_data[kind_data['price'] > 0]
        info['brand_amount'] = len(kind_data)
        info['brand_price_max'] = kind_data['price'].max()
        info['brand_price_median'] = kind_data['price'].median()
        info['brand_price_min'] = kind_data['price'].min()
        info['brand_price_sum'] = kind_data['price'].sum()
        info['brand_price_std'] = kind_data['price'].std()
        info['brand_price_average'] = round(kind_data['price'].sum() / len(kind_data) + 1, 2)  # 小数点后两位
        all_info[kind] = info

    pd.DataFrame(all_info).T.reset_index()
    brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})  # 将index列名修改为brand
    test_data = test_data.merge(brand_fe, how='left', on='brand')

    # 数据分桶，以power为例
    binary = [i * 10 for i in range(31)]
    test_data['power_binary'] = pd.cut(test_data['power'], binary, labels=False)
    test_data = test_data.drop(['creatDate', 'regDate', 'regionCode'], axis=1)

    # 压缩空间
    test_data = reduce_mem_usage(test_data)

    feature_names = [x for x in test_data.columns if x not in ['brand', 'model', 'regDate', 'creatDate', 'price']]
    test_data = test_data.dropna().replace('-', 0)
    test_data['notRepairedDamage'] = test_data['notRepairedDamage'].astype(np.float32)
    test_data_x = test_data[feature_names]
    test_data['price'] = model.predict(test_data_x)

    # 保存

    predict['price'] = test_data['price']
    predict = predict.fillna(0)
    sava_predict(predict)


# 保存预测数据
def sava_predict(data):
    data.to_csv("../prediction_result/predictions.csv", index=0)


if __name__ == '__main__':
    # 训练数据准备
    train_dataset = reduce_mem_usage(load_data())

    # 得到相关训练数据
    train_dataset, train_dataset_x, train_dataset_y, deal_feature_names = prepare_data(train_dataset)

    # 构建简单的线性模型
    get_model = simple_linear_model(train_dataset, train_dataset_x, train_dataset_y, deal_feature_names)

    # 测试
    test_dataset = pd.read_csv('../data/used_car_testB_20200421.csv', sep=" ")  # 得到测试数据
    predict_y(test_dataset, get_model)
