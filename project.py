import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import re
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_eigenvalues(G):
    pca = PCA().fit(G.values)
    eigenvalues = pca.explained_variance_
    print(eigenvalues)
    plt.plot(range(len(eigenvalues)), eigenvalues)
    plt.title("Sorted Eigenvalues After Running PCA on All SNPs")
    plt.xlabel("Eigenvalue Number")
    plt.ylabel("Eigenvalue Value")
    plt.show()


def generate_data(G, phen_file):
    pca = PCA(n_components=50).fit_transform(G.values)
    snps_df = pd.DataFrame(pca)
    snps_df['iid'] = G['iid']

    y = pd.read_csv(phen_file, sep=" ", header=None, usecols=[1, 2], names=['iid', 'y'])
    snps_df = snps_df.merge(y, how='inner', on='iid').drop(columns=['iid'])

    model = os.path.normpath(phen_file).split(os.sep)[1]
    hsv = re.search("[0-9]\.[0-9]\.hsq", phen_file).group(0)[:3]
    rep = phen_file[-6]
    train_df, test_df = train_test_split(snps_df, test_size=0.25)
    train_file = "train_{}_{}_{}.csv".format(model, hsv, rep)
    test_file = "test_{}_{}_{}.csv".format(model, hsv, rep)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print("Wrote training data to:", train_file)
    print("Wrote testing data to:", test_file)


def get_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.loc[:, test_df.columns != 'y']
    y_train = train_df['y']
    X_test = test_df.loc[:, test_df.columns != 'y']
    y_test = test_df['y']

    return X_train.values, y_train.values, X_test.values, y_test.values


def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Linear Regression MSE: ", mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)


def ridge_regression(X_train, y_train, X_test, y_test, alpha):
    model = Ridge(alpha=alpha).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Ridge Regression MSE: ", mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)


def elasticnet_regression(X_train, y_train, X_test, y_test, alpha):
    model = ElasticNet(alpha=alpha).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("ElasticNet Regression MSE: ", mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)


def mlp_regression(X_train, y_train, X_test, y_test):
    model = MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate='adaptive', max_iter=1000).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("MLP Regression MSE: ", mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)


def neural_net_tf(X_train, y_train, X_test, y_test, flag):
    tf.random.set_seed(0)
    MODEL_FLAG = flag

    if MODEL_FLAG == 'FC':
        model = tf.keras.Sequential([
            Input(shape=(50,)),
            Dense(50, activation='relu'),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(10, activation='relu'),
            Dense(1)
        ])
    elif MODEL_FLAG == 'FC_DROP':
        model = tf.keras.Sequential([
            Input(shape=(50,)),
            Dense(50, activation='relu'),
            Dropout(rate=0.5),
            Dense(50, activation='relu'),
            Dropout(rate=0.5),
            Dense(25, activation='relu'),
            Dropout(rate=0.5),
            Dense(10, activation='relu'),
            Dropout(rate=0.5),
            Dense(1)
        ])
    elif MODEL_FLAG == 'CNN_SIMPLE':
        model = tf.keras.Sequential([
            Conv1D(20, kernel_size=4, activation='relu', input_shape=(50, 1)),
            Dropout(rate=0.5),
            Flatten(),
            Dense(25, activation='relu'),
            Dropout(rate=0.5),
            Dense(10, activation='relu'),
            Dropout(rate=0.5),
            Dense(1)
        ])
    elif MODEL_FLAG == 'CNN':
        model = tf.keras.Sequential([
            Conv1D(20, kernel_size=4, activation='relu', input_shape=(50, 1)),
            Dropout(rate=0.5),
            MaxPooling1D(pool_size=4, strides=1),
            Conv1D(20, kernel_size=4, activation='relu'),
            Dropout(rate=0.5),
            MaxPooling1D(pool_size=4, strides=1),
            Conv1D(20, kernel_size=4, activation='relu'),
            Dropout(rate=0.5),
            Flatten(),
            Dense(25, activation='relu'),
            Dropout(rate=0.5),
            Dense(10, activation='relu'),
            Dropout(rate=0.5),
            Dense(1)
        ])
    elif MODEL_FLAG == 'LSTM':
        model = tf.keras.Sequential([
            LSTM(4, input_shape=(50, 1)),
            Dropout(rate=0.5),
            Dense(10, activation='relu'),
            Dropout(rate=0.5),
            Dense(1)
        ])
    elif MODEL_FLAG == 'GRU':
        model = tf.keras.Sequential([
            GRU(4, input_shape=(50, 1)),
            Dropout(rate=0.5),
            Dense(10, activation='relu'),
            Dropout(rate=0.5),
            Dense(1)
        ])

    model.compile(optimizer=Adam(), loss=tf.keras.losses.MeanSquaredError())
    # model.summary()

    EPOCHS = 250
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=0)
    y_pred = model.predict(X_test)
    # print("Neural Net MSE: ", mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred)


def run_all_models(X_train, y_train, X_test, y_test):
    linear = linear_regression(X_train, y_train, X_test, y_test)
    ridge = ridge_regression(X_train, y_train, X_test, y_test, alpha=25)
    elasticnet = elasticnet_regression(X_train, y_train, X_test, y_test, alpha=25)
    # mlp_regression(X_train, y_train, X_test, y_test)

    neural_fc = []
    neural_fc_drop = []
    neural_cnn_simple = []
    neural_cnn = []
    for i in range(1):
        neural_fc.append(neural_net_tf(X_train, y_train, X_test, y_test, 'FC'))
        neural_fc_drop.append(neural_net_tf(X_train, y_train, X_test, y_test, 'FC_DROP'))
        neural_cnn_simple.append(neural_net_tf(X_train, y_train, X_test, y_test, 'CNN_SIMPLE'))
        neural_cnn.append(neural_net_tf(X_train, y_train, X_test, y_test, 'CNN'))
    return linear, ridge, elasticnet, np.mean(neural_fc), np.mean(neural_fc_drop), np.mean(neural_cnn_simple), np.mean(neural_cnn)


if __name__ == "__main__":
    G = read_plink1_bin("Genotypes//sample.bed", "Genotypes//sample.bim", "Genotypes//sample.fam", verbose=False)
    # plot_eigenvalues(G)

    # # run this once
    # hsqs = [0.0, 0.2, 0.4, 0.8]
    # for hsq in hsqs:
    #     filename = "Phenotypes//OnePercCont//Sim.OnePercCausal.{}.hsq.Continuous.1.phen".format(hsq)
    #     generate_data(G, filename)

    # # hyperparameter optimization
    # X_train, y_train, X_test, y_test = get_data("train_InfCont_0.4_1.csv", "test_InfCont_0.4_1.csv")
    # # normalize between -1 and 1
    # y_min = np.min(np.concatenate((y_train, y_test)))
    # y_max = np.max(np.concatenate((y_train, y_test)))
    # y_train = 2*(y_train-y_min) / (y_max - y_min) - 1
    # y_test = 2*(y_test-y_min) / (y_max - y_min) - 1
    # print("Linear Regression:", linear_regression(X_train, y_train, X_test, y_test))
    # print("Ridge Regression:", ridge_regression(X_train, y_train, X_test, y_test, alpha=25))
    # print("Elastic Net:", elasticnet_regression(X_train, y_train, X_test, y_test, alpha=25))
    # print("FC DROP:", neural_net_tf(X_train, y_train, X_test, y_test, flag="FC_DROP"))
    # print("CNN SIMPLE:", neural_net_tf(X_train, y_train, X_test, y_test, flag="CNN_SIMPLE"))
    # print("CNN:", neural_net_tf(X_train, y_train, X_test, y_test, flag="CNN"))
    # print("LSTM:", neural_net_tf(X_train, y_train, X_test, y_test, flag="LSTM"))
    # print("GRU:", neural_net_tf(X_train, y_train, X_test, y_test, flag="GRU"))

    models = ["InfCont", "OnePercCont"]
    model = "InfCont"
    hsqs = [0.0, 0.2, 0.4, 0.8]
    # results = []
    # for hsq in hsqs:
    #     print("Testing hsq={}".format(hsq))
    #     X_train, y_train, X_test, y_test = get_data("train_{}_{}_1.csv".format(model, hsq), "test_{}_{}_1.csv".format(model, hsq))
    #     # normalize between -1 and 1
    #     y_min = np.min(np.concatenate((y_train, y_test)))
    #     y_max = np.max(np.concatenate((y_train, y_test)))
    #     y_train = 2*(y_train-y_min) / (y_max - y_min) - 1
    #     y_test = 2*(y_test-y_min) / (y_max - y_min) - 1
    #     # run all models
    #     result = run_all_models(X_train, y_train, X_test, y_test)
    #     results.append(list(result))
    # print(results)

    if model == "InfCont":
        # Infinitesimal
        results = [[0.157873, 0.157853, 0.133297, 0.500796, 0.131613, 0.133292, 0.133084],
                   [0.127768, 0.127758, 0.125987, 0.471380, 0.125559, 0.125993, 0.125977],
                   [0.115064, 0.115047, 0.115039, 0.746204, 0.115610, 0.115043, 0.115017],
                   [0.111210, 0.111199, 0.110114, 0.414749, 0.109985, 0.110115, 0.110097]]
        results_df = pd.DataFrame(np.array(results).T,
                                  columns=["hsq=0.0", "hsq=0.2", "hsq=0.4", "hsq=0.8"],
                                  index=["Linear", "Ridge", "ElasticNet", "FC", "FC_DROP", "CNN_SIMPLE", "CNN"])
        results_df.to_csv("InfCount_mse.csv")
        print(results_df)
        ax = results_df.plot(kind="bar",
                             title="MSE of Each Model over $h^2$ Values on the Infinitesimal Dataset",
                             rot=0,
                             legend=True,
                             width=0.8)
        for p in ax.patches:
            ax.annotate(np.round(p.get_height(), decimals=3),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 15),
                        textcoords='offset points',
                        rotation=90,
                        fontsize=8)
        ax.set_xticklabels(["Linear", "Ridge", "ElasticNet", "FC", "FC_DROP", "CNN_SIMPLE", "CNN"], fontsize=8)
        ax.set_xlabel("hsq ($h^2$)")
        ax.set_ylabel("MSE")
        ax.set_ylim(0, np.max(results)+0.1)
        plt.show()

    elif model == "OnePercCont":
        # One Percent
        results = [[0.1246450345610211, 0.12463272130881987, 0.1132847447749337, 0.7019090215040166, 0.11374954513120553, 0.113235719379747, 0.11325743751760942],
                   [0.08733213309969404, 0.08730848582503088, 0.06542295547602535, 0.4854621421874476, 0.0669825550334337, 0.06543594059691678, 0.06543397583275107],
                   [0.11606670881453982, 0.11605181114258172, 0.10643575272844816, 0.48155801029270656, 0.10572504437994643, 0.106507218400322, 0.1064901006028444],
                   [0.10024739614288698, 0.10023483499015717, 0.09931725938135898, 0.2890996986180322, 0.10002563603873912, 0.09936247389587273, 0.09932543456154799]]
        results_df = pd.DataFrame(np.array(results).T,
                                  columns=["hsq=0.0", "hsq=0.2", "hsq=0.4", "hsq=0.8"],
                                  index=["Linear", "Ridge", "ElasticNet", "FC", "FC_DROP", "CNN_SIMPLE", "CNN"])
        results_df.to_csv("OnePercent_mse.csv")
        print(results_df)
        ax = results_df.plot(kind="bar",
                             title="MSE of Each Model over $h^2$ Values on the One Percent Dataset",
                             rot=0,
                             legend=True,
                             width=0.8)
        for p in ax.patches:
            ax.annotate(np.round(p.get_height(), decimals=3),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 15),
                        textcoords='offset points',
                        rotation=90,
                        fontsize=8)
        ax.set_xticklabels(["Linear", "Ridge", "ElasticNet", "FC", "FC_DROP", "CNN_SIMPLE", "CNN"], fontsize=8)
        ax.set_xlabel("hsq ($h^2$)")
        ax.set_ylabel("MSE")
        ax.set_ylim(0, np.max(results)+0.1)
        plt.show()



    # X = np.arange((len(results[0]))*5, step=5)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    # fig.subplots_adjust(hspace=0.01)
    #
    # ax2.set_xlabel("hsq ($H^2$)")
    # ax2.set_ylabel("MSE")
    # ax2.set_xticks(X+1.5)
    # ax2.set_yticks([0, 1, 2, 3, 4])
    # ax2.set_ylim(0, 5)  # outliers only
    # if model == "InfCont":
    #     ax1.set_title("MSE of Each Model Over $hsq$ Values on the Infinitesimal Dataset")
    #     ax1.set_ylim(10000, 700000)  # most of the data
    # elif model == "OnePercCont":
    #     ax1.set_title("MSE of Each Model Over $hsq$ Values on the One Percent Dataset")
    #     ax1.set_ylim(500, 6000)  # most of the data
    # ax1.spines.bottom.set_visible(False)
    # ax2.spines.top.set_visible(False)
    # ax1.xaxis.tick_top()
    # ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    # ax2.xaxis.tick_bottom()
    # ax2.set_xticklabels(["Linear", "Ridge", "ElasticNet", "FC", "FC_DROP", "CNN_SIMPLE", "CNN"], fontsize=8)
    # ax1.bar(X + 0, results[0], width=1, align='center', label="hsq=0.0")
    # ax1.bar(X + 1, results[1], width=1, align='center', label="hsq=0.2")
    # ax1.bar(X + 2, results[2], width=1, align='center', label="hsq=0.4")
    # ax1.bar(X + 3, results[3], width=1, align='center', label="hsq=0.8")
    # ax1.bar(X + 4, [0]*len(results[0]), width=1, align='center')
    # ax2.bar(X + 0, results[0], width=1, align='center', label="hsq=0.0")
    # ax2.bar(X + 1, results[1], width=1, align='center', label="hsq=0.2")
    # ax2.bar(X + 2, results[2], width=1, align='center', label="hsq=0.4")
    # ax2.bar(X + 3, results[3], width=1, align='center', label="hsq=0.8")
    # ax2.bar(X + 4, [0]*len(results[0]), width=1, align='center')
    # ax1.legend()
    # d = .5  # proportion of vertical to horizontal extent of the slanted line
    # kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
    #               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    # # ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    # ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    # plt.show()