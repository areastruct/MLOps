"""
This is a boilerplate pipeline 'clust_demo'
generated using Kedro 0.18.2
"""
import logging.config
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score #,silhouette_score
from collections import Counter

logger = logging.getLogger(__name__)


def scaler(df):
    return StandardScaler().fit_transform(df)


def getNormTransform(data, norm_type="zscore"):
    """
    norm_type:
        zscore: return index of best values
        standard: returen after StandardScaler

    # for n in ['zscore', 'standard', 'quantile', 'log_norm']:
    #     if n == 'zscore':
    #         sns.distplot(df['M_CHECK_SUM'].iloc[getNormTransform(df['M_CHECK_SUM'], norm_type = n)])
    #     else:
    #         sns.distplot(getNormTransform(df['M_CHECK_SUM'], norm_type = n))
    #     plt.show()
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import quantile_transform

    if norm_type == 'zscore':
        zscore = (data - np.mean(data)) / np.std(data)
        return np.where(abs(zscore) <= 3)[0]

    if norm_type == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(data.to_frame())

    if norm_type == 'quantile':
        return quantile_transform(data.to_frame(), output_distribution='normal')

    if norm_type == 'log_norm':
        return np.log(abs(data - np.mean(data)) / np.std(data)).replace([np.inf, -np.inf], np.nan)


def pcator(df, cols):
    tmp = scaler(df[cols].fillna(0))
    pca = PCA(n_components=3)
    pca.fit(tmp)
    # подготовка DF для обучения
    pca_df = pca.transform(df[cols].fillna(0))

    return pca_df
def prep_df(df, cols, meth):
    tmp = pd.DataFrame(index=df.index)

    if meth != 'pca':

        for n in cols:
            if meth == 'zscore':
                res = getNormTransform(df[n], norm_type=meth)
                tmp[n] = np.nan
                tmp[n].iloc[res] = df[n].iloc[res]

            if meth == 'quantile':
                tmp[n] = getNormTransform(df[n], norm_type=meth).reshape(1, -1)[0]

            if meth == 'box-cox':
                tmp[n] = boxcox1p(df[n], 0)

            if meth == 'standard':
                tmp[n] = getNormTransform(df[n], norm_type=meth).reshape(1, -1)[0]

            if meth == 'wo':
                tmp[n] = df[n]

    else:
        tmp = pcator(df, cols)

    return tmp


def calcClusterFact(clear_df, k=3, viz_cols=None, viz=True):
    if viz_cols is not None:
        z = viz_cols[0]
        o = viz_cols[1]
        viz_cols = [(list(clear_df.columns).index(z), list(clear_df.columns).index(o)), ]
    else:
        viz_cols = [(0, 1), ]

    print('---------------------------------------------------------------------------')
    # обучаем модель на всех данных
    model = KMeans(n_clusters=k, random_state=777)
    model.fit(clear_df)
    print(model)

    # результат + количество по каждому кластеру
    predict_labels = model.labels_
    print(Counter(predict_labels))

    # рассчет метрик
    calinski_harabasz_score_metric = calinski_harabasz_score(clear_df, predict_labels)
    davies_bouldin_score_metric = davies_bouldin_score(clear_df, predict_labels)
    print('calinski_harabasz_score:', calinski_harabasz_score_metric)
    print('davies_bouldin_score:', davies_bouldin_score_metric)

    # визуализация кластерво по всем осям
    # if viz:
    #     for i, j in viz_cols:
    #
    #         cmap = plt.cm.get_cmap('jet')
    #         plt.figure(figsize=(9, 9))
    #
    #         for cluster in range(k):
    #             sns.scatterplot(clear_df.iloc[np.where(predict_labels == cluster)[0], i],
    #                             clear_df.iloc[np.where(predict_labels == cluster)[0], j], color=cmap(cluster * 100))
    #         plt.show()
    # print('---------------------------------------------------------------------------')

    return predict_labels, davies_bouldin_score_metric

def first():
    cols = [('set 1', ['CS1_PRC', 'CS2_PRC', 'CS3_PRC', 'CS4_PRC']), ]
    meths = ['pca', ]

    df = pd.DataFrame(abs(np.random.randn(100, 4)) / 10, columns=['CS1_PRC', 'CS2_PRC', 'CS3_PRC', 'CS4_PRC'])
    df['CUS'] = np.random.randint(1, 999, size=(100, 1))
    df = df[['CUS', 'CS1_PRC', 'CS2_PRC', 'CS3_PRC', 'CS4_PRC']]
    # df.head()
    clear_df = df.copy()

    print('Preprocessing')
    print('\n')
    print('Generating features...')

    # fig, ax = plt.subplots(1, 4, figsize=(15, 3))
    # sns.distplot(clear_df['CS1_PRC'], ax=ax[0])
    # sns.distplot(clear_df['CS2_PRC'], ax=ax[1])
    # sns.distplot(clear_df['CS3_PRC'], ax=ax[2])
    # sns.distplot(clear_df['CS4_PRC'], ax=ax[3])
    # plt.show()

    metrics = list()
    for st, cls in cols:
        for meth in meths:
            pre_df = pd.DataFrame(scaler(clear_df[cls]), columns=cls)
            print('Learning ...')

            print('=======>', st, meth, '<=======')
            if meth != 'pca':
                tmp = prep_df(pre_df, cls, meth).fillna(0)
            else:
                tmp = prep_df(pre_df, cls, meth)
            print(tmp.shape)

            predict_labels, db_metric = calcClusterFact(tmp, k=3, viz=False)
            X = clear_df.copy()[['CUS', ] + cls].fillna(0)
            X['cluster'] = predict_labels
            metrics.append(db_metric)

            # # формируем компоненты
            # pca = PCA(n_components=2)
            # X['x'] = pca.fit_transform(X[X.columns[1:]])[:, 0]
            # X['y'] = pca.fit_transform(X[X.columns[1:]])[:, 1]
            #
            # # для построения визуализации
            # customer_clusters = X[['cluster', 'x', 'y']]
            # color_labels = customer_clusters['cluster'].unique()
            # rgb_values = sns.color_palette("Set2", 8)
            # color_map = dict(zip(color_labels, rgb_values))
            # plt.subplots(figsize=(4, 4))
            # plt.scatter(customer_clusters['x'], customer_clusters['y'], c=customer_clusters['cluster'].map(color_map))
            # plt.ylabel('Y-principal component 2')
            # plt.xlabel('X-principal component 1')
            # print(davies_bouldin_score(X[X.columns[1:-3]], predict_labels))
            # plt.show()
            #
            # for i in cls:
            #     sns.boxplot(y=clear_df[i], x=X['cluster'], data=X, orient="v", showfliers=False)
            #     plt.show()

            print('=======   =======   =======   =======')
            print('AVG SCORE -', sum(metrics) / len(metrics))
            print('\n')

            usrs = pd.DataFrame()
            usrs['CUS'] = clear_df['CUS']
            usrs['SEG'] = X['cluster']

            # usrs.to_csv('seg_{}_{}_{}_new23.csv'.format(en, st, meth), sep=',', header = True, index=False)
            usrs.head()
    return True
