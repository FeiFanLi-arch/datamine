from sklearn import preprocessing
from deal_data import *
warnings.filterwarnings('ignore')


# 删除训练集异常值
def delete_abnormal(train_data, train_cols_name, scale=3):

    # 用于清洗异常值，默认使用box_plot（scale=3）进行清洗
    def boxplot_delete(train_data_ser, box_scale):
        iqr = box_scale * (train_data_ser.quantile(0.75) - train_data_ser.quantile(0.25))
        val_low = train_data_ser.quantile(0.25) - iqr
        val_up = train_data_ser.quantile(0.75) + iqr
        rule_low = (train_data_ser < val_low)
        rule_up = (train_data_ser > val_up)

        return (rule_low, rule_up), (val_low, val_up)

    train_data_n = train_data.copy()
    train_data_series = train_data_n[train_cols_name]
    rule, value = boxplot_delete(train_data_series, box_scale=scale)
    index = np.arange(train_data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is：{}".format(len(index)))
    train_data_n = train_data_n.drop(index)
    train_data_n.reset_index(drop=True, inplace=True)
    print("Now column number is：{}".format(train_data_n.shape[0]))
    index_low = np.arange(train_data_series.shape[0])[rule[0]]
    deletes = train_data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(deletes).describe())
    index_up = np.arange(train_data_series.shape[0])[rule[1]]
    deletes = train_data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(deletes).describe())

    # 可视化箱型图
    look_delete_boxplot(train_data, train_data_n, train_cols_name)

    return train_data_n


# 可视化清洗的箱型图
def look_delete_boxplot(train_data, train_data_n, train_cols_name):
    sns.boxplot(data=train_data, y=train_data[train_cols_name], palette='Set1')
    plt.show()
    sns.boxplot(data=train_data_n, y=train_data_n[train_cols_name], palette='Set1')
    plt.show()


# 构造特证
def create_feature_structure(train_data, test_data):
    # 将训练集和测试集放在一起进行构造
    train_data['train'] = 1
    test_data['train'] = 1
    data = pd.concat([train_data, test_data], ignore_index=True, sort=False)

    # 构造使用时间,Y必须大写
    car_use_time = pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce')
    car_reg_time = pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')
    data['used_time'] = (car_use_time - car_reg_time).dt.days

    # 查看使用时间中空数据
    null_time_data = data['used_time'].isnull().sum()
    print(null_time_data)

    # 从邮编中提取城市信息
    # data['city'] = data['regionCode'].apply(lambda x: str(x)[:-3])

    # 计算某个品牌的销售量
    train_data_gb = train_data.groupby("brand")  # 根据brand对数组进行分类
    all_info = {}
    for kind, kind_data in train_data_gb:
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
    data = data.merge(brand_fe, how='left', on='brand')
    print(data)

    # 数据分桶，以power为例
    binary = [i * 10 for i in range(31)]
    data['power_binary'] = pd.cut(data['power'], binary, labels=False)

    # 删除原始数据
    data = data.drop(['creatDate', 'regDate', 'regionCode'], axis=1)
    print(data.columns)

    # 导出给树模型
    save_data_for_tree(data)

    # 数据分桶举例
    # power
    data = power_binary(data, train_data)

    # kilometer
    kilometer_binary(data)

    # 归一化
    for feature in data.columns:
        if feature.startswith('brand'):
            data[feature] = max_min(data[feature])

    # 对类别特征进行OneEncoder
    data = pd.get_dummies(data, columns=['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage',
                                         'power_binary'])
    print(data.shape)
    print(data.columns)

    # 导出给lr（逻辑回归）模型
    save_data_for_lr(data)


# 保存用于树模型数据
def save_data_for_tree(data):
    data.to_csv('../userdata/data_for_tree.csv', index=0)


# 保存用于lr模型数据
def save_data_for_lr(data):
    data.to_csv('../userdata/data_for_lr.csv', index=0)


# power数据分桶
def power_binary(data, train_data):
    # 查看data_power数据分布
    data['power'].plot.hist()
    plt.show()
    # 分布不好

    # 查看训练集power数据分布
    train_data['power'].plot.hist()
    plt.show()

    # 对data_power取对数，并进行归一化
    preprocessing.MinMaxScaler()
    data['power'] = np.log(data['power'] + 1)
    data['power'] = ((data['power'] - np.min(data['power'])) / (np.max(data['power']) - np.min(data['power'])))
    data['power'].plot.hist()
    plt.show()

    return data


# kilometer数据分桶
def kilometer_binary(data):
    data['kilometer'].plot.hist()
    plt.show()
    # 由可视化图可知，数据分布较好

    # 归一化
    data['kilometer'] = ((data['kilometer'] - np.min(data['kilometer'])) / (
                np.max(data['kilometer']) - np.min(data['kilometer'])))
    data['kilometer'].plot.hist()
    plt.show()


# 定义归一化函数
def max_min(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


if __name__ == '__main__':
    # 得到数据
    train_dataset, test_dataset = load_dataset()

    # 查看数据集尺寸
    look_datasize(train_dataset, test_dataset)

    # 查看数据结构
    look_dataset_structure(train_dataset, test_dataset)

    # 查看数据列名
    look_data_cols(train_dataset, test_dataset)

    # 删除异常数据，以power为例
    train_dataset = delete_abnormal(train_dataset, 'power', scale=3)

    # 特证构造
    create_feature_structure(train_dataset, test_dataset)
