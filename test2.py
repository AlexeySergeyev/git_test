import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import umap
# from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn import preprocessing



def griz_show(df):
    xedges = np.linspace(0, 1, 100)
    yedges = np.linspace(-0.5, 0.5, 100)

    x = df['psfMag_g'] - df['psfMag_r']
    y = df['psfMag_i'] - df['psfMag_z']

    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    plt.imshow(np.log10(H + 1), interpolation='bilinear', origin='low', cmap='jet',
               # norm=LogNorm(vmin=0.1, vmax=10000),
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
               )

    cls = pd.read_csv(f'{path}class_colors.csv')
    for i in range(cls.__len__()):
        plt.scatter(cls.iloc[i]['g-r'], cls.iloc[i]['i-z'], marker='.', color='black')
        plt.annotate(cls.iloc[i]['class'],
                     xy=(cls.iloc[i]['g-r'] + 0.01, cls.iloc[i]['i-z'] + 0.01))

    plt.show()


def color_combinations(df):
    cls = pd.read_csv(f'{path}class_colors.csv')

    key0 = cls.keys()[0]
    print(key0)
    print(df.__len__())

    df_dist = pd.DataFrame()
    for i in range(cls.__len__()):
        # key = cls.keys()[i]
        dist = np.abs(df['psfMag_g'] - df['psfMag_u'] - cls['g-u'][i]) + \
            np.abs(df['psfMag_g'] - df['psfMag_r'] - cls['g-r'][i]) + \
            np.abs(df['psfMag_g'] - df['psfMag_i'] - cls['g-i'][i]) + \
            np.abs(df['psfMag_i'] - df['psfMag_z'] - cls['i-z'][i])
        df_dist[cls['class'][i]] = dist

    min = pd.DataFrame(df_dist.idxmin(axis=1), columns=['class'])
    df['class'] = df_dist.idxmin(axis=1)
    print(min['class'].value_counts())
    # print(df_dist)
    # print(min)
    min['color'] = min['class'].apply(ord)-65

    # print(min)
    # min['color'] = ord
    # print(cls['class'][min])
    df['u-g'] = df['psfMag_u'] - df['psfMag_g']
    df['u-r'] = df['psfMag_u'] - df['psfMag_r']
    df['u-i'] = df['psfMag_u'] - df['psfMag_i']
    df['u-z'] = df['psfMag_u'] - df['psfMag_z']

    df['g-r'] = df['psfMag_g'] - df['psfMag_r']
    df['g-i'] = df['psfMag_g'] - df['psfMag_i']
    df['g-z'] = df['psfMag_g'] - df['psfMag_z']

    df['r-i'] = df['psfMag_r'] - df['psfMag_i']
    df['r-z'] = df['psfMag_r'] - df['psfMag_z']

    df['i-z'] = df['psfMag_i'] - df['psfMag_z']

    import matplotlib.cm as cm

    # colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    # plt.scatter(df['g-r'], df['i-z'], marker='.', c=min['color'],
    #             cmap='Set3', alpha=0.5, edgecolors='none')

    colors_values = list(set(min['color']))
    colors_names = cls['class'].to_list()
    colors = dict(zip(colors_names, colors_values))
    # print(colors_values, colors_names)
    # for area in [100, 300, 500]:
    #     plt.scatter([], [], c='k', alpha=0.3, s=area,
    #                 label=str(area) + ' km$^2$')
    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')

    # cmap = plt.cm.Greys
    # cmin = np.min(colors_values)
    # import matplotlib.colors
    # norm = matplotlib.colors.Normalize(vmin=np.min(colors_values), vmax=np.max(colors_values))
    # for i, val in enumerate(colors_values):
    #     plt.scatter([], [], c=cmap(colors_values[i]), alpha=1,
    #                 label=colors_names[i], norm=norm)
    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1)

    # plt.legend(colors_values, colors_names)
    # plt.legend()

    # for i in range(cls.__len__()):
    #     plt.scatter(cls.iloc[i]['g-r'], cls.iloc[i]['i-z'], marker='.', color='black')
    #     plt.annotate(cls.iloc[i]['class'],
    #                  xy=(cls.iloc[i]['g-r'] + 0.01, cls.iloc[i]['i-z'] + 0.01), weight='bold')
    #     sns.jointplot(x="x", y="y", data=df, kind="kde");

    # N = 11
    # for i in range(N):
    #     sns.palplot(sns.light_palette(sns.color_palette()[0], i+1))
    # print(len(sns.color_palette()))
    # plt.show()
    # for x in df['class'].map(d_cls):
    #     print(x)
    # c = [sns.color_palette()[x] for x in df['class'].map(d_cls)]
    # print(c)
    # plt.show()

    sns.pairplot(df[['u-g', 'u-r', 'u-i', 'u-z',
                     'g-r', 'g-i', 'g-z',
                     'r-i', 'r-z',
                     'i-z', 'class']], hue='class')
    plt.show()


def show_umap(df):
    reducer = umap.UMAP(n_neighbors=50,
                        min_dist=0.01,
                        n_components=4,
                        metric='euclidean')

    new_data = df[['u-g', 'u-r', 'u-i', 'u-z',
                   'g-r', 'g-i', 'g-z',
                   'r-i', 'r-z',
                   'i-z']].values
    scaled_new_data = StandardScaler().fit_transform(new_data)
    embedding = reducer.fit_transform(scaled_new_data)
    print(embedding.shape)

    # for x in df['class'].map(d_cls):
    #     print(x, end=',')
    print(set(df['class']))
    labels = list(set(df['class']))

    a = sns.palplot(sns.color_palette('Spectral', 11))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette('Paired')[x] for x in df['class'].map(d_cls)],
    marker='.')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Penguin dataset', fontsize=24)
    umap.plot.points(reducer, labels=labels)
    plt.show()

    # fig, ax = plt.subplots(1, figsize=(14, 10))
    # plt.scatter(*embedding.T, s=0.1, c=target, cmap='Spectral', alpha=1.0)
    # plt.setp(ax, xticks=[], yticks=[])
    # cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
    # cbar.set_ticks(np.arange(10))
    # cbar.set_ticklabels(classes)
    # plt.show()


def load_class_from_spectra():
    df = pd.read_csv(f'{path}sdss_from_spectra.csv')
    print(df)
    # print(df[df['complex'].isna() == True])
    # print(df['complex'].value_counts())
    df = df.dropna(subset=['complex'])
    print(df['complex'].value_counts())

    # print(set(df['name']))
    benoit_names = set(df['name'])
    print('asteroids spectra:', len(benoit_names))
    spec = df.groupby('name', as_index=False).first()
    # print(spec)

    sdss = pd.read_csv(f'{path}sso_tot4c.csv')
    cond = sdss['bcolor_ru'] & sdss['bcolor_rg'] & sdss['bcolor_ri'] & sdss['bcolor_rz'] & \
           sdss['bastrom_u'] & sdss['bastrom_g'] & sdss['bastrom_r'] & sdss['bastrom_i'] & \
           sdss['bastrom_z'] & sdss['bphot_u'] & sdss['bphot_g'] & sdss['bphot_r'] & \
           sdss['bphot_i'] & sdss['bphot_z']

    sdss = sdss[cond]
    coincidences = sdss[sdss['Name'].isin(benoit_names)]

    # sdss_names = list(set(coincidences['Name']))
    # print(sdss_names)
    # print(coincidences['Name'])
    print('asteroids sdss:', len(set(coincidences['Name'])))

    classes = np.zeros(coincidences.__len__(), dtype=(np.str))
    for i in range(coincidences.__len__()):
        obj = coincidences.iloc[i]['Name']
        val = spec.loc[spec['name'] == obj]['complex'].values[0]
        classes[i] = val
    coincidences.loc[:, 'complex'] = classes
    # print(coincidences)

    dictarr = []
    for key in d_cls:
        # dictarr.append(d_cls[key])
        dictarr.append(key)

    d_stat = pd.DataFrame(d_cls.items(), columns=['complex', 'id'])
    del d_stat['id']
    # cond = coincidences.loc['class'] == d_stat.loc['class']
    # d_stat.loc['total'] = pd.Series(coincidences[cond].sum())

    coincidences.loc[:, 'u-g'] = coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_g']
    coincidences.loc[:, 'u-r'] = coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_r']
    coincidences.loc[:, 'u-i'] = coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_i']
    coincidences.loc[:, 'u-z'] = coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_z']
    coincidences.loc[:, 'g-r'] = coincidences.loc[:, 'psfMag_g'] - coincidences.loc[:, 'psfMag_r']
    coincidences.loc[:, 'g-i'] = coincidences.loc[:, 'psfMag_g'] - coincidences.loc[:, 'psfMag_i']
    coincidences.loc[:, 'g-z'] = coincidences.loc[:, 'psfMag_g'] - coincidences.loc[:, 'psfMag_z']
    coincidences.loc[:, 'r-i'] = coincidences.loc[:, 'psfMag_r'] - coincidences.loc[:, 'psfMag_i']
    coincidences.loc[:, 'r-z'] = coincidences.loc[:, 'psfMag_r'] - coincidences.loc[:, 'psfMag_z']
    coincidences.loc[:, 'i-z'] = coincidences.loc[:, 'psfMag_i'] - coincidences.loc[:, 'psfMag_z']

    ast_group = coincidences.groupby('complex')
    print(ast_group['complex'].count())
    d_stat = d_stat.join(ast_group['complex'].count(), on='complex', rsuffix='_r')
    d_stat = d_stat.join(ast_group['u-g'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['u-r'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['u-i'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['u-z'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['g-r'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['g-i'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['g-z'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['r-i'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['r-z'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['i-z'].mean(), on='complex', rsuffix='_mean')

    d_stat = d_stat.join(ast_group['u-g'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['u-r'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['u-i'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['u-z'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['g-r'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['g-i'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['g-z'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['r-i'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['r-z'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['i-z'].std(), on='complex', rsuffix='_std')


    # d_stat = d_stat.join(ast_group['u-g'].median(), on='complex', lsuffix='median_')

    # d_stat = d_stat.rename({'u-g_r': ''})
    # d_stat = d_stat.join(ast_group['u-r'].mean(), on='complex', lsuffix='mean_')
    # d_stat = d_stat.join(ast_group['u-r'].median(), on='complex', lsuffix='median_')
    d_stat = d_stat.rename(columns={'complex_r': 'total', 'u-g': 'mean_u-g', 'u-r': 'mean_u-r'})

    pd.set_option("display.precision", 3)
    print(d_stat)
    # # d_stat = pd.merge(d_stat, ast_group['complex'].count(), left_on='complex', right_on='complex')
    # # print(d_stat)
    #
    # tmp = np.zeros(d_stat.__len__())
    # for i in range(d_stat.__len__()):
    #     cond = coincidences['complex'] == d_stat.iloc[i]['complex']
    #     tmp[i] = len(coincidences[cond])
    # d_stat['total'] = tmp
    # print(d_stat)

    print(coincidences['complex'].value_counts())
    isshow = False
    if isshow:
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})
        pal = sns.color_palette('Paired', 11)
        cond_c = (coincidences['complex'] == 'A') | (coincidences['complex'] == 'K') | \
                 (coincidences['complex'] == 'L') | (coincidences['complex'] == 'Q') | \
                 (coincidences['complex'] == 'B') | (coincidences['complex'] == 'D')
        cond_c = ~cond_c
        x = coincidences['psfMag_g'][cond_c] - coincidences['psfMag_r'][cond_c]
        y = coincidences['psfMag_i'][cond_c] - coincidences['psfMag_z'][cond_c]
        ax1.scatter(x=x, y=y,
                    c=[pal[x] for x in coincidences['complex'][cond_c].map(d_cls)],
                    marker='.')
        ax2 = sns.barplot(y=dictarr, x=np.ones(len(dictarr)), palette=pal)
        ax2.set_xticks([])
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

        ax1.set_xlabel('g-r')
        ax1.set_ylabel('i-z')
        ax1.set_xlim([0.1, 1.0])
        ax1.set_ylim([-0.7, 0.7])

        # plt.savefig('./figs/spec_class1.png', dpi=300)
        plt.show()

    issave = True
    if issave:
        coincidences.to_csv('.\data\ccoincidences.csv', index=False)

    return coincidences


def knn_class():
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score, cross_val_predict

    coincidences = pd.read_csv('.\data\ccoincidences.csv')
    X = coincidences[color_list]
    y = coincidences['complex']

    my_scaler = preprocessing.StandardScaler()
    my_scaler.fit(X)
    X_2 = my_scaler.transform(X)

    a_tot = coincidences['complex'].value_counts()

    for state in range(0, 1):
        X_train, X_test, y_train, y_test = \
            train_test_split(X_2, y, stratify=y, random_state=state)

        all_scores = []
        n_neighb = np.arange(10, 13)
        for n_neighbors in n_neighb:
            print(n_neighbors)
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
            knn_clf.fit(X_train, y_train)
            # knn_clf.score(X_train, y_train)
            # print(knn_clf.score(X_train, y_train))
            # knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=3)

            Z = cross_val_predict(knn_clf, X, y, cv=3)
            cond = (Z != y)

            all_scores.append((n_neighbors,
                               # knn_scores.mean(),
                               # knn_scores.std(),
                               coincidences.loc[:, ['complex']][cond].__len__()
                               ))
            wrong = coincidences.loc[:, ['complex']][cond]
            a1 = wrong['complex'].value_counts()
            print((a_tot - a1)/ a_tot)
            # print(coincidences['complex'].value_counts())

        # res = np.array(sorted(all_scores, key=lambda x: x[1], reverse=True))
        all_scores = np.array(all_scores)
        # for item in res:
        #     print(item)
        # Z = knn_clf.predict(X)
        # cond = (Z != y)
        # df_wrong = coincidences.loc[:, ['complex']][cond]
        # print(df_wrong.__len__())

        # my_scaler = preprocessing.StandardScaler()
        # my_scaler.fit(X)
        # X_2 = my_scaler.transform(X)
        # # X_2 = preprocessing.scale(X)
        # print(X_2.std(axis=0))
        #
        # neighb = np.arange(2, 50)
        # # print(neighb)
        # score1 = np.zeros(neighb.shape[0])
        # for i, n_neighbors in enumerate(neighb):
        #     # we create an instance of Neighbours Classifier and fit the data.
        #     clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform',n_jobs=4)
        #     clf.fit(X, y)
        #     Z = clf.predict(X)
        #     cond = (Z != y)
        #     df_wrong = coincidences.loc[:, ['complex']][cond]
        #     score1[i] = df_wrong.__len__()
        #     print(i, end=', ')
        #
        # print()
        # score2 = np.zeros(neighb.shape[0])
        # for i, n_neighbors in enumerate(neighb):
        #     # we create an instance of Neighbours Classifier and fit the data.
        #     clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', n_jobs=4)
        #     clf.fit(X_2, y)
        #     Z = clf.predict(X_2)
        #     cond = (Z != y)
        #     df_wrong = coincidences.loc[:, ['complex']][cond]
        #     score2[i] = df_wrong.__len__()
        #     print(i, end=', ')
        #
        isshow = True
        if isshow:
            plt.ylabel('wrong calss')
            plt.xlabel('neighbors')

            plt.plot(n_neighb, all_scores[:, 1], label='X')
            # plt.plot(n_neighb, all_scores[:, 1], label='X')
            # plt.errorbar(x=n_neighb, y=all_scores[:, 1], yerr=all_scores[:, 2])

    plt.legend()
    plt.show()
        #
    # # cond = (Z != y)
    # # print(coincidences.loc[:, ['complex']][cond])
    # # print(test_score)





if __name__ == '__main__':
    path = f'./data/'
    d_cls = {"A": 0, "B": 1, "C": 2, "D": 3, "K": 4, "L": 5, "Q": 6, "S": 7, "U": 8, "V": 9, "X": 10}
    color_list = ['u-g', 'u-r', 'u-i', 'u-z',
                  'g-r', 'g-i', 'g-z',
                  'r-i', 'r-z',
                  'i-z']
    # print(d_cls.items())

    # df = pd.read_csv(f'{path}last/sso_tot4c.csv')
    # print('Load complete.')
    # cond = df['bcolor_ru'] & df['bcolor_rg'] & df['bcolor_ri'] & df['bcolor_rz'] & \
    #     df['bastrom_u'] & df['bastrom_g'] & df['bastrom_r'] & df['bastrom_i'] & df['bastrom_z']
    # df = df[['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z']][cond].reset_index()
    # griz_show()
    # color_combinations()
    # coinc = load_class_from_spectra()
    knn_class()



