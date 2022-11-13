import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msn
import scipy.stats as st
import pandas_profiling
import warnings
warnings.filterwarnings('ignore')


# 加载数据集
def load_dataset():
    path = "../data"
    train_data = pd.read_csv(path + "/used_car_train_20200313.csv", sep=" ")
    test_data = pd.read_csv(path + "/used_car_testB_20200421.csv", sep=" ")
    return train_data, test_data


# 查看数据集尺寸
def look_datasize(train_data, test_data):
    print("train", train_data.shape)
    print("test", test_data.shape)


# 浏览数据集构造
def look_dataset_structure(train_data, test_data):
    train_data_head = train_data.head(10)
    test_data_head = test_data.head(10)
    print(train_data_head)
    print(test_data_head)


# 查看数据列名
def look_data_cols(train_data, test_data):
    train_cols = train_data.columns
    test_cols = test_data.columns
    print(train_cols)
    print(test_cols)


# 查看缺失值
def look_miss_data(train_data, test_data):
    train_data_miss = train_data.isnull().sum()
    test_data_miss = test_data.isnull().sum()
    print(train_data_miss)
    print(test_data_miss)
    # 可视化处理
    loss_data_bar(train_data_miss, test_data_miss)


# 缺失值条状图
def loss_data_bar(train_data_miss, test_data_miss):
    # 训练集可视化
    plt.figure(1)
    plt.title("train_dataset_miss")
    train_data_miss = train_data_miss[train_data_miss > 0]
    train_data_miss.sort_values(inplace=True)
    train_data_miss.plot.bar()
    plt.show()

    # 测试集可视化
    plt.figure(2)
    plt.title("test_dataset_miss")
    test_data_miss = test_data_miss[test_data_miss > 0]
    test_data_miss.sort_values(inplace=True)
    test_data_miss.plot.bar()
    plt.show()


# 查看缺省值矩阵图
def look_default_data_matrix(train_data, test_data):
    # 训练集
    msn.matrix(train_data.sample(250))
    plt.show()

    # 测试集
    msn.matrix(test_data.sample(250))
    plt.show()


# 查看缺省值条状图
def look_default_data_bar(train_data, test_data):
    # 训练集
    msn.bar(train_data.sample(1000))
    plt.show()

    # 测试集
    msn.bar(test_data.sample(1000))
    plt.show()


# 异常值
def abnormal_data(train_data, test_data):
    # 查看异常值检测
    list_train_data = train_data.info()
    list_test_data = test_data.info()
    print(list_train_data)
    print(list_test_data)

    # 显示不同类型值
    different_train_datatype = train_data['notRepairedDamage'].value_counts()
    different_test_datatype = test_data['notRepairedDamage'].value_counts()
    print(different_train_datatype)
    print(different_test_datatype)

    # 存在缺省值，替换为nan
    train_data["notRepairedDamage"].replace('-', np.nan, inplace=True)
    test_data["notRepairedDamage"].replace('-', np.nan, inplace=True)

    # 再次查看，确认是否替换成功
    different_train_datatype = train_data['notRepairedDamage'].value_counts()
    different_test_datatype = test_data['notRepairedDamage'].value_counts()
    print(different_train_datatype)
    print(different_test_datatype)

    # 再次查看缺失值状况
    train_data_miss = train_data.isnull().sum()
    test_data_miss = test_data.isnull().sum()
    print(train_data_miss)
    print(test_data_miss)

    # 查看预测值分布
    train_predict = train_data['price']
    print(train_predict)
    train_predicts = train_data['price'].value_counts()
    print(train_predicts)

    return train_data, test_data


# 训练集总体分布情况可视化图
def train_total_distribution(train_data):
    y = train_data['price']
    plt.figure(1)
    plt.title('Johnson SU')
    sns.distplot(y, kde=False, fit=st.johnsonsu)
    plt.show()
    plt.figure(2)
    plt.title('Normal')
    sns.distplot(y, kde=False, fit=st.norm)
    plt.show()
    plt.figure(3)
    plt.title('Log Normal')
    sns.distplot(y, kde=False, fit=st.lognorm)
    plt.show()


# 训练集价格的可视化图，及其偏度（Skewness）和峰度（Kurtosis）
def train_price(train_data):
    sns.distplot(train_data['price'])
    print("Skewness：%f" % train_data['price'].skew())
    print("Kurtosis：%f" % train_data['price'].kurt())
    plt.show()


# 训练集偏度、峰度并进行可视化
def train_total(train_data):
    train_skewness = train_data.skew()
    train_kurtosis = train_data.kurt()
    print(train_skewness)
    print(train_kurtosis)
    sns.distplot(train_skewness, color="blue", axlabel="Skewness")
    plt.show()
    sns.distplot(train_kurtosis, color="red", axlabel="Kurtosis")
    plt.show()


# 预测值频数可视化图
def look_predict_frequency(train_data):
    plt.hist(train_data['price'], color="blue")
    plt.show()


# 训练集的预测值进行log变换，并进行可视化
def train_transformer(train_data):
    trans_data = np.log(train_data['price'])
    plt.hist(trans_data, color="blue")
    plt.show()


# 将特征划分为数字特征和类别特征
def train_classifier():
    # 下面方法适用于没有直接label coding的数据
    # numeric_features = train_data.select_dtypes(include=[np.number]).columns
    # categorical_features = train_data.select_dtypes(include=[np.object]).columns
    # print(numeric_features)
    # print(categorical_features)
    # 手动进行划分数字特征和类别特征
    numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2',
                        'v_3', 'v_4', 'v_5', 'v_6', 'v_7',
                        'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
    categorical_features = ['name', 'model', 'brand', 'bodyType',
                            'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']
    return numeric_features, categorical_features


# 数字特征相关分析
def numeric_analyse(numb_features, train_data):
    # 将预测值加入数字特征中
    # numeric_features = numb_features.append('price')  # 该条语句错误
    numeric_features = numb_features
    numeric_features.append('price')

    # 相关性分析
    price_numeric = train_data[numeric_features]
    # 对数字特征进行线性分析
    correlations = price_numeric.corr()
    # 提取数字特征中的价格特征，并降序排序
    price_correlation = correlations['price'].sort_values(ascending=False)
    print(price_correlation)

    # 可视化处理
    numeric_features_analyse(correlations)
    del price_numeric['price']

    # 数字特征偏度与峰度分析
    look_numeric_feature_property(numeric_features, train_data)

    # 数字特征分布
    numeric_features_distribution(numeric_features, train_data)

    # 相关数字特征之间关系
    numeric_features_relation(train_data)


# 数字特征相关性分析可视化
def numeric_features_analyse(correlation):
    plt.subplots(figsize=(7, 7))
    plt.title("Correlation of Numeric Feature with Price", y=1, size=14)
    sns.heatmap(correlation, square=True, vmax=0.8)
    plt.show()


# 查看数字特征偏度与峰度
def look_numeric_feature_property(numb_features, train_data):
    for numeric_feature in numb_features:
        print("特征：", numeric_feature)
        print("Skewness：{:15}".format(train_data[numeric_feature].skew()))
        print("Kurtosis：{:06.2f}".format(train_data[numeric_feature].kurt()))


# 数字特征分布可视化
def numeric_features_distribution(numb_features, train_data):
    f = pd.melt(train_data, value_vars=numb_features)
    print(f)
    # 在一个图上显示
    sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False).map(sns.distplot, "value")
    plt.show()
    # 由图可知：匿名特征分布的较为均匀


# 数字特征之间的关系可视化
def numeric_features_relation(train_data):
    sns.set()
    columns = ['price', 'v_12', 'v_8', 'v_0', 'power', 'v_5', 'v_2', 'v_6', 'v_1', 'v_14']
    sns.pairplot(train_data[columns], size=2, kind='scatter', diag_kind='kde')
    plt.show()


# 类别特征相关分析
def look_categorical_distribution(cate_features, train_data):
    # 查看类别特征的unique分布
    for categorical_feature in cate_features:
        print(categorical_feature + "的特征分布如下：")
        print("{}特征分布有{}个不同的值".format(categorical_feature, train_data[categorical_feature].nunique()))
        print(train_data[categorical_feature].value_counts())

    # name和regionCode过于稀疏，不进行画图
    new_categorical_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for categorical_feature in new_categorical_features:
        train_data[categorical_feature] = train_data[categorical_feature].astype('category')
        if train_data[categorical_feature].isnull().any():
            train_data[categorical_feature] = train_data[categorical_feature].cat.add_categories(['MISSING'])
            train_data[categorical_feature] = train_data[categorical_feature].fillna('MISSING')

    # 箱型图
    categorical_features_box(train_data, new_categorical_features)

    # 小提琴图
    categorical_features_violin(train_data, new_categorical_features)

    # 柱状图
    categorical_features_bar(train_data, new_categorical_features)

    # 频数图
    categorical_features_frequency(train_data, new_categorical_features)

    return train_data


# 类别特征箱型图可视化
def categorical_features_box(train_data, categorical_features):
    def boxplot(x, y, **kwargs):
        sns.boxplot(x=x, y=y)
        plt.xticks(rotation=90)

    f = pd.melt(train_data, id_vars=['price'], value_vars=categorical_features)
    print(f)
    sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False).map(boxplot, "value", "price")
    plt.show()


# 类别特征小提琴图
def categorical_features_violin(train_data, categorical_features):
    for categorical_feature in categorical_features:
        sns.violinplot(train_data, x=categorical_feature, y='price')
        plt.show()


# 类别特征柱状图
def categorical_features_bar(train_data, categorical_features):
    def bar_plot(x, y, **kwargs):
        sns.barplot(x=x, y=y)
        plt.xticks(rotation=90)

    f = pd.melt(train_data, id_vars=['price'], value_vars=categorical_features)
    sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False).map(bar_plot, "value", "price")
    plt.show()


# 类别特征频数图
def categorical_features_frequency(train_data, categorical_features):
    def count_plot(x, **kwargs):
        sns.countplot(x=x)
        plt.xticks(rotation=90)

    f = pd.melt(train_data, value_vars=categorical_features)
    sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False).map(count_plot, "value")
    plt.show()


def generate_data_report(train_data):
    pfr = pandas_profiling.ProfileReport(train_data)
    pfr.to_file("../userdata/data_analyse.html")


if __name__ == '__main__':
    # 获取数据
    train_dataset, test_dataset = load_dataset()
    # 查看数据集尺寸
    look_datasize(train_dataset, test_dataset)

    # 简单浏览数据集的构造
    look_dataset_structure(train_dataset, test_dataset)

    # 数据集特证名
    look_data_cols(train_dataset, test_dataset)

    # 查看缺失值
    look_miss_data(train_dataset, test_dataset)

    # 查看缺省值矩阵图
    look_default_data_matrix(train_dataset, test_dataset)

    # 查看缺省条状图
    look_default_data_bar(train_dataset, test_dataset)

    # 查看异常值
    train_dataset, test_dataset = abnormal_data(train_dataset, test_dataset)

    # 训练集总体分布情况
    train_total_distribution(train_dataset)
    # 由可视化图可知，价格的分布更加符合无界约翰逊分布，不符合正态分布，故回归之前要对数据进行处理

    # 训练集price偏度、峰度情况
    train_price(train_dataset)

    # 训练集偏度、峰度
    train_total(train_dataset)

    # 查看预测值（price）的具体频数
    look_predict_frequency(train_dataset)
    # 查看可视化图可知，大于20000的值很少，可以将这部分数据当做异常值填充或处理掉

    # 对训练集的预测值进行log变换
    train_transformer(train_dataset)
    # 经过转换后，分布的更加均匀

    # 对特征进行划分
    num_features, cat_features = train_classifier()

    # 数字特征分析
    numeric_analyse(num_features, train_dataset)

    # 类别特征
    train_dataset = look_categorical_distribution(cat_features, train_dataset)

    # 生成数据报告
    generate_data_report(train_dataset)
