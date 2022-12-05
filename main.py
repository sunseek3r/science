from tslearn.datasets import UCR_UEA_datasets
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
import numpy


def draw_plot(X, y_pred, model):
    plt.figure()
    for i in range(2):
        plt.subplot(2, 1, 1 + i)
        for x in X[y_pred == i]:
            plt.plot(x.ravel(), 'k-', alpha=.2)
        plt.plot(model.cluster_centers_[i].ravel(), 'r-')
        plt.title('Cluster %d' % (i + 1))
    plt.tight_layout()
    plt.show()


def compare(y_pred, y_train, n_ts):
    y_compare1 = []
    y_compare2 = []
    for i in y_pred:
        if i == 0:
            y_compare1.append(2)
            y_compare2.append(1)
        else:
            y_compare1.append(1)
            y_compare2.append(2)
    score1, score2 = 0, 0
    for i, j, k in zip(y_train, y_compare1, y_compare2):
        if i == j:
            score1 += 1
        if i == k:
            score2 += 1
    print(score1 / float(n_ts), score2 / float(n_ts))


def dtw_test():
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('Strawberry')
    print("create model")
    model = TimeSeriesKMeans(n_clusters=2, metric='dtw',
                             max_iter=10, random_state=seed)
    print("train start")
    model.fit(X_train)
    print(X_train.shape)
    transfer(model)


def kshape_test():
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('Strawberry')
    model = KShape(n_clusters=2, verbose=True, random_state=seed)
    y_pred = model.fit_predict(X_train)
    draw_plot(X_train, y_pred, model)


def kernel_test():
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('Strawberry')
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    gak = KernelKMeans(n_clusters=2, kernel='gak', kernel_params={'sigma': 'auto'},
                       n_init=20, verbose=True, random_state=seed, n_jobs=-1)
    y_pred = gak.fit_predict(X_train)
    n_ts, sz, d = X_train.shape
    compare(y_pred, y_train, n_ts)


def transfer(global_model):
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('Wine')
    local_model = TimeSeriesKMeans(n_clusters=2, metric='dtw',
                                   max_iter=10, random_state=seed)
    global_model.fit(X_train)
    y_pred_test_global = global_model.predict(X_test[:10])
    y_pred_test_local = local_model.fit_predict(X_test[:10])

    draw_plot(X_test[:10], y_pred_test_global, global_model)
    draw_plot(X_test[:10], y_pred_test_local, local_model)


if __name__ == '__main__':
    dtw_test()
