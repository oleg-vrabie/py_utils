# #############################################################################
#   Useful functions for collecting, representing, separating into classes and
# clustering ICP time-series.

# New functions will be added at the begining, just after imports.
# #############################################################################
#                                  Imports
# #############################################################################
import numpy as np
import hdbscan          # Hierarchichal DBSCAN
from sklearn import mixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import os       # For getcdw() in plot_means_a4
# For NMF
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
import random

default_colors = ['navy', 'c', 'cornflowerblue', 'gold', 'dimgray', 'darkorange',
                  'magenta', 'saddlebrown', 'forestgreen', 'turquoise', 'black',
                  'darkkhaki', 'crimson', 'sienna', 'peru', 'slategray', 'olive']

# #############################################################################
#                                  Functions
# #############################################################################

def gaussian_mixture_pca_projections(X, waves, n_components,
                                     npat=None, scid=None, wave_type=None,
                                     max_iter=500, bayesian=True, full=True,
                                     with_mahal=False,
                                     additional_clustering=False,
                                     colors=default_colors,
                                     separate_smallest_component=False,
                                     rotate=False):
# =========================================================================
# Clustering of ICP data projected onto first three principal components,
# using Gaussian Mixture Models
#
# Inputs:
#   X := (float) aray with shape (nr_waves, 3), containing projections of nr_waves-waves
#        onto first three principal components
#   waves := (float) array of shape (nr_waves, wave_size=780), containing single
#            ICP waves as rows
#   n_components := (int) nr of components parameter fed into GaussianMixture models
#                   algorithm, i.e. number of Gaussian components data is factorized into
#   max_iter := (int) maximum nr of iterations allowed for GMM fit
#   bayesian := (bool) 'True' for BayesianGaussianMixture
# =========================================================================
# Outputs:
#   clust_mean_waves := (nr_waves, wave_size) array containing clusters' means
#   clusters := list of arrays, each comprising individual clusters identified
#   ns := list of cluster indexes that needed additional_clustering
#   ks := list of numbers of components that "ns" clusters were splitted into
# =========================================================================

    if bayesian==True:
        print('Fitting BayesianGaussianMixture using {} components\n'.format(n_components))
        gmm = mixture.BayesianGaussianMixture(n_components=n_components,
                                              covariance_type='full',
                                              max_iter=max_iter,
                                              n_init=10,
                                              verbose=1).fit(X)
    else:
        print('Fitting GaussianMixture using {} components\n'.format(n_components))
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full',
                                      max_iter=max_iter,
                                      n_init=10,
                                      verbose=1).fit(X)



    # Collecting necessary quantities:
    clusters_list = []      # list of arrays of points belonging to each clusters
    index_list = []         # list of array of indexes of waves in each cluster
    xs_list = []            # lists containing coordinates of points in each cluster
    ys_list = []
    zs_list = []
    cov_list = []           # list of covariance matrices for each cluster
    centroids_list = []     # centroids' positions for each cluster
    mahal_list = []         # list of Mahalanobis distances for each cluster
    score_samples_list = [] # weighted log probabilities for each point in the cluster
    clust_mean_waves = np.zeros((n_components, 780))    # rows are mean waves coresponding to clusters' centroids
    waves_by_cluster = [] # list of clusters with length=n_clusters and
                                # shape (n_waves_in_cluster, wave_size=780)

    print('\nCollecting centroids:')
    predictions = gmm.predict(X)
    for i in range(n_components):
        index_list.append(np.where(i == predictions)[0])
        clust_mean_waves[i] = np.mean(waves[i==predictions], axis=0)
        waves_by_cluster.append(np.asarray(waves[i==predictions, :]))
        clusters_list.append(np.asarray(X[i==predictions]))
        cov_list.append(MinCovDet().fit(clusters_list[i]))
        score_samples_list.append(np.asarray(gmm.score_samples(X)))
        cluster_i = np.reshape(clusters_list[i], (clusters_list[i].shape[0] ,3))
        centroids_list.append(gmm.means_[i])
        if with_mahal == True:
            print('\n\t... and Mahalanobis distances:')
            mahal_list.append(cdist(np.reshape(gmm.means_[i], (1, -1)), cluster_i, metric='mahalanobis'))

        # Collecting individual coordinates of points of each cluster
        xs_list.append(cluster_i[:, 0])
        ys_list.append(cluster_i[:, 1])
        zs_list.append(cluster_i[:, 2])
        print('\t{}/{} of shape {}'.format(i+1, n_components, cluster_i.shape))

    print('\nClusters (components): {}'.format(len(clusters_list)))
    assert len(clusters_list) == n_components


    #print('\nwaves_by_cluster: {}'.format(waves_by_cluster))
    #print('\nwaves_by_cluster[0].shape: {}'.format(waves_by_cluster[0].shape))
    #print('\nwaves_by_cluster: {}'.format(len(waves_by_cluster[0])))
    #print('\nwaves_by_cluster: {}'.format(len(waves_by_cluster[1])))


    # =========================================================================
    # Separating the smallest component from the rest of projections
    # =========================================================================
    # Usually, the biggest variance along PC3 belongs to the smallest and most
    # noisiest component. It will be useful to separate it to see robust
    # features represented in this dimension.
    # In the case of PAT7, nsc, the smallest component (~100 elements) gives the
    # variance along PC3.
    if separate_smallest_component == True:
        count_list = []
        for i in range(n_components):
            print('{} {}'.format(i, clusters_list[i].shape))
            count_list.append(clusters_list[i].shape[0])

        print(count_list)
        clusters_list.pop(count_list.index(min(count_list)))

        for i in range(len(clusters_list)):
            print('{} {}'.format(i, clusters_list[i].shape))

        n_components = n_components - 1

    # =========================================================================
    # Plotting/animating projected data with markers==cluster index
    # =========================================================================
    # Visualization parameters:

    fig = plt.figure(num='Projection',figsize=(16,9))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.3)
    ax = fig.add_subplot(111, projection='3d')

    for i in range(n_components):
        xs = xs_list[i]
        ys = ys_list[i]
        zs = zs_list[i]


        print('{} added'.format(i+1))

        ax.scatter(xs, ys, zs,
                   c=colors[i],
                   marker="${}$".format(i+1),
                   depthshade=False,
                   s=40,
                   label='{} elements'.format(xs.shape[0]))

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.title('Projection of original {}({}) data onto first three PCs (NFPAT{})'
              .format(wave_type, scid, npat))

    # rotate the axes and update
    if rotate == True:
        plt.ion()
        print('Rotating entire projection...')
        for angle in range(0, 360, 12):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.000001)

    plt.show(block=False)
    plt.pause(0.001)
    #plt.ioff()
    # =========================================================================
    # Plotting separate clusters with/and Mahalanobis distances
    # =========================================================================
    plot_separate_clusters(n_components, xs_list, ys_list, zs_list,
                           centroids_list, mahal_list, with_mahal=with_mahal)
    plt.show(block=False)
    plt.pause(0.001)
    # =========================================================================
    # Visualize means of clusters = coresponding "types" of waves
    # =========================================================================
    plot_means_of_clusters(n_components, waves, clust_mean_waves, colors, 'Initial means')
    #plot_means_of_clusters(n_means, waves, clust_mean_waves, colors, ')
    #plt.draw()
    #plt.show()
    plt.show(block=False)
    plt.pause(0.001)

    # =========================================================================
    # Additional clustering (if needed)
    # =========================================================================
    clusters = waves_by_cluster

    verdict = input('Additional clustering? (yes, y, 1): ')
    if verdict == 'yes': #or 'y' or '1':
        ns = []
        ks = []
        x, add_clusters, n, k, index_list = additional_clustering_(clusters_list,
                                                       waves_by_cluster,
                                                       index_list,
                                                       xs_list, ys_list, zs_list,
                                                       scid, colors)

        clust_mean_waves = np.concatenate((clust_mean_waves, x), axis=0)
        #clusters.pop(n)   # Remove initial cluster "n"
        clusters.extend(add_clusters)
        ns.append(n)
        ks.append(k)
        plt.show(block=False)
        plt.pause(0.001)

        while input('Still? (yes : True; Enter : False): ')=='yes':
            x, add_clusters, n, k, index_list = additional_clustering_(clusters_list,
                                                                waves_by_cluster,
                                                                index_list,
                                                                xs_list, ys_list, zs_list,
                                                                scid, colors)
            ns.append(n)
            ks.append(k)
            clust_mean_waves = np.concatenate((clust_mean_waves, x), axis=0)
            #clusters.pop(n)   # Remove initial cluster "n"
            clusters.extend(add_clusters)
            plt.show(block=False)
            plt.pause(0.001)
    else:
        ns = None
        ks = 0

    return clust_mean_waves, clusters, ns, ks, index_list
# #############################################################################
# #############################################################################
def additional_clustering_(clusters_list, waves_by_cluster, index_list,
                           xs_list, ys_list, zs_list, scid, my_colors=default_colors):
    print('\nAdditional clustering')
    gmm_component, k = list(map(int, input('Give the component and the nr of clusters to split in (separated by comma): ')
                                           .split(',')))
    component = np.reshape(clusters_list[gmm_component-1],
                           (clusters_list[gmm_component-1].shape[0] ,3))
    component_waves = waves_by_cluster[gmm_component-1]

    # =========================================================================
    # Clustering step
    # =========================================================================
    type = input('k, dbscan or gmm: ')
    if type == 'k':
        model = KMeans(n_clusters=k,
                       n_init=10).fit(component)
        sub_colors = model.labels_.ravel()
        index_list.pop(gmm_component-1)
        index_list.append(sub_colors)

        x_centro = model.cluster_centers_[:, 0]
        y_centro = model.cluster_centers_[:, 1]
        z_centro = model.cluster_centers_[:, 2]

    if type == 'gmm':
        max_iter = 500
        model = mixture.BayesianGaussianMixture(n_components=k,
                                                covariance_type='full',
                                                max_iter=max_iter,
                                                n_init=10,
                                                verbose=1).fit(component)
        sub_colors = model.predict(component).ravel()

        x_centro = model.means_[:, 0]
        y_centro = model.means_[:, 1]
        z_centro = model.means_[:, 2]

    if type == 'dbscan':
        print('\n\tAttention! Experimental! Error prone!\n')
        #min_cluster_size = int(input('min_cluster_size: '))
        min_cluster_size = int(input('min_cluster_size: '))
        min_samples = int(input('min_samples: '))
        #model = hdbscan.HDBSCAN(min_cluster_size=min_samples)
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples).fit(component)
        #labels = model.fit_predict(component)
        labels = model.labels_
        print('components_ {}'.format(labels.max()+1))
        k = labels.max()+1
        sub_colors = labels.ravel()
        index_list.pop(gmm_component-1)
        index_list.append(sub_colors)

    # =========================================================================
    # Indexing
    # =========================================================================
    index_list.pop(gmm_component-1)

    # Implement a function for this block (for loop)    TODO
    for i in range(k):
        index_list.append(np.where(i == sub_colors)[0])

    # =========================================================================
    # Plot separate subclusters with different colors
    # =========================================================================
    xs = xs_list[gmm_component-1]
    ys = ys_list[gmm_component-1]
    zs = zs_list[gmm_component-1]

    # Coordinates of the centroid:
    #x_centro, y_centro, z_centro = centroids_list[gmm_component-1]
    fig = plt.figure(num='Component {}'.format(gmm_component), figsize=(16,9))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs,
               c=sub_colors,
               cmap=matplotlib.colors.ListedColormap(default_colors, N=len(np.unique(sub_colors))),
               depthshade=False,
               s=1)
    # Add centoids
    if type != 'dbscan':
        ax.scatter(x_centro, y_centro, z_centro,
                   marker="X",
                   s=300,
                   c='green',
                   label='Centroids')

    ax.set_xlabel('PC1 ({})'.format(gmm_component))
    ax.set_ylabel('PC2 ({})'.format(gmm_component))
    ax.set_zlabel('PC3 ({})'.format(gmm_component))
    plt.title('Separate component #{} into {} clusters ({} elements)'.format(gmm_component, k, xs.shape[0]))
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)

    # =========================================================================
    # Plot means of separate subclusters:
    # =========================================================================
    additional_mean_waves = np.zeros((k, 780))
    additional_clusters_list = []       # list with waves from additional clusters

    if type != 'dbscan':
        print('\nAdditionally extracted (shapes):')
        for i in range(k):
            print(component_waves[i==model.predict(component), :].shape)
            additional_clusters_list.append(component_waves[i==model.predict(component), :])
            additional_mean_waves[i] = np.mean(component_waves[i==model.predict(component), :], axis=0)
        title = 'Means from the separated component #{} ({})'.format(gmm_component, scid)

        plot_means_of_clusters(k, component_waves, additional_mean_waves,
                               colors=np.unique(sub_colors))
    else:
        print('\nAdditionally extracted (shapes):')
        for i in range(k):
            print(component_waves[i==labels].shape)
            additional_clusters_list.append(component_waves[i==labels, :])
            additional_mean_waves[i] = np.mean(component_waves[i==labels], axis=0)
        title = 'Means from the separated component #{} ({})'.format(gmm_component, scid)

        plot_means_of_clusters(k, component_waves, additional_mean_waves,
                               colors=np.unique(sub_colors))

    return additional_mean_waves, additional_clusters_list, gmm_component-1, k, index_list
# #############################################################################
# #############################################################################

def plot_means_of_clusters(n_means, waves, clust_mean_waves, colors=default_colors, title='Means'):
    if n_means <= 2:
        side1 = 1
    else:
        side1 = n_means//3

    if side1 == 1:
        top = 0.75
        bottom = 0.25
    else:
        top = None
        bottom = None
    side2 = n_means//side1

    if side1*side2 < n_means:
        side2 += 1
    wave_size = waves.shape[1]  #780

    """
    if n_means > len(colors):
        difference = n_means-len(colors)
        print('\nNote! {} additional (navy) colors were added!'.format(difference))
        for i in range(difference):
            colors.append('navy')
    """
    print('Plottin means of clusters...')

    fig, ax = plt.subplots(num='{} means'.format(n_means), figsize=(16,9))
    fig.subplots_adjust(left=0.1, right=0.9, top=top, bottom=bottom, hspace=0.245)

    for i in range(n_means):
        if isinstance(colors[i], str) == True:
            color = colors[i]
        else:
            color = default_colors[colors[i]]
        plt.subplot(side1, side2, i+1)
        plt.plot(np.arange(0, wave_size, 1), clust_mean_waves[i], color=color)
        plt.title('{}'.format(i+1))
    plt.suptitle(title)
    #plt.show(block=False)
    #plt.pause(0.001)

    return
# #############################################################################
# #############################################################################

def plot_means_a4(cluster_means, color='navy', title='Means', path=os.getcwd()):
    n_means = len(cluster_means)
    wave_size = cluster_means[0].shape[0]  #780
    pwd = path + '/'

    side1 = 5   # n rows
    side2 = 2   # n columns

    top = 0.940
    bottom = 0.035

    fig, ax = plt.subplots(num='{} means'.format(n_means), figsize=(8.27,11.69))
    fig.subplots_adjust(left=0.1, right=0.9, top=top, bottom=bottom, hspace=0.365)

    for i in range(n_means):
        plt.subplot(side1, side2, i+1)
        plt.plot(np.arange(0, wave_size, 1), cluster_means[i], color=color)
        plt.title('{}'.format(i+1))
    plt.suptitle(title)
    plt.savefig(pwd+str(input('Give a filename to save as pdf: ')))
    plt.show()

    return fig
# #############################################################################
# #############################################################################

def plot_random_wave_samples(concatenated_waves, wave_type='icp', seg_type='scp',
                             k=20, color='navy', rand_list=[], path=os.getcwd()):
    # Plots a sequence of unique samples selected from concatenated_waves
    #print(rand_list)
    if not rand_list.size:  # Check if empty
        print('Sampling a random list of {} elements'.format(k))
        rand_list = random.sample(range(concatenated_waves.shape[0]), k)
    else:
        rand_list = rand_list[:k]
    fig, ax = plt.subplots(num='{} means'.format(k), figsize=(16,9))
    fig.subplots_adjust(left=0.1, right=0.9, hspace=0.365)

    for i in range(len(rand_list)):
        wave = concatenated_waves[rand_list[i], :]
        plt.subplot(4, 5, i+1)
        plt.plot(np.arange(0, wave.shape[0], 1), wave, color=color)
        plt.title('{}'.format(rand_list[i]))
    plt.suptitle('Random {} ({}) samples'.format(wave_type, seg_type))
    #plt.savefig(pwd+str(input('Give a filename to save as pdf: ')))
    plt.show()

    return rand_list
# #############################################################################
# #############################################################################

def plot_separate_clusters(n_clusters, xs_list, ys_list, zs_list, centroids_list,
                           mahal_list, with_mahal=True, colors=default_colors):
    print('\nPlotting separate clusters:\n')

    for i in range(n_clusters):
        # Coordinates of points in the cluster
        xs = xs_list[i]
        ys = ys_list[i]
        zs = zs_list[i]
        # Coordinates of the centroid:
        x_centro, y_centro, z_centro = centroids_list[i]

        # Markers' color:
        if with_mahal == True:      # colormap = inferno
            mahal_colors = mahal_list[i].ravel()
            edge=None
            face=None
        else:       # empty circles
            mahal_colors = 'none'
            edge='navy'
            face='none'

        fig = plt.figure(num='Cluster {}'.format(i+1), figsize=(16,9))
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(xs, ys, zs,
                       c=mahal_colors,
                       edgecolors=edge,
                       facecolors=face,
                       cmap=plt.cm.inferno,
                       depthshade=False,
                       s=1)
        # Add centoids
        ax.scatter(x_centro, y_centro, z_centro,
                   marker="X",
                   s=300,
                   color='green',
                   label='Centroid')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        if with_mahal == True:
            plt.colorbar(s).set_label('Mahalanobis distance')
        plt.title('Cluster #{}/{} ({} elements)'.format(i+1, n_clusters, xs.shape[0]))
        plt.legend()
    return

# #############################################################################
# #############################################################################
def nmf_plot_by_component(n_components, X, scid, init='nndsvda', verbose=1, max_iter=500):
    nmf = NMF(n_components=n_components, init='nndsvda', verbose=1, max_iter=500)
    #nmf.fit(scp_waves)
    print('\n\tNMF with k={} of {} segments'.format(n_components, scid))

    X[X < 0] = 0
    W = nmf.fit_transform(X)
    H = nmf.components_

    H = normalize(H, norm='max')
    print(H.shape)

    side1 = 2
    side2 = 3
    wave_size = 780

    for i in range(n_components):
        plt.subplot(side1, side2, i+1)
        plt.plot(H[i], c=default_colors[i])

    plt.suptitle('NMF results')
    plt.show()

# #############################################################################
# #############################################################################
def cut_by_size_in2_classes(a_waves, b_waves, wave_size):
    # Cut waves from raw a- and b_waves pulses of wave_size and label them
    nr_a = len(a_waves)//wave_size
    nr_b = len(b_waves)//wave_size

    nr_waves = nr_a + nr_b

    waves = np.zeros((nr_waves, wave_size))
    labels = np.zeros((nr_waves,), dtype=int)

    for i in range(nr_waves):
        start = i*wave_size
        stop = start + wave_size
        if i < nr_b:
            labels[i] = 1


            for j in range(start,stop):
                index = j - start
                waves[i][index] = b_waves[j]
        else:
            start = i*wave_size - nr_b*wave_size
            stop = start + wave_size
            labels[i] = 0
            for j in range(start,stop):
                index = j - start
                waves[i][index] = a_waves[j]

    return waves, labels

# #############################################################################
# #############################################################################
def pca_projections(X, n_components, svd_solver='auto'):
    from sklearn.decomposition import PCA as PCA

    pca = PCA(n_components)
    fitted = pca.fit_transform(X)

    return fitted
# #############################################################################
# #############################################################################
def plot_3d(xs, ys, zs,
            window_title='New Figure',
            title='3D plot',
            xlabel='x',
            ylabel='y',
            zlabel='z',
            s=60):

    fig = plt.figure(num=window_title,figsize=(16,9))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.3)
    ax = fig.add_subplot(111, projection='3d')

    edge='navy'
    face='none'

    ax.scatter(xs, ys, zs,
               depthshade=False,
               edgecolors=edge,
               facecolors=face,
               s=s,
               label='{} elements'.format(xs.shape[0]))

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.title(title)
    #plt.show()
    plt.show(block=False)
    plt.pause(0.001)

    return

# #############################################################################
# #############################################################################
def plot_two_by_two(xs1, ys1, zs1,
                    xs2, ys2, zs2,
                    values3,
                    values4,
                    window_title='', suptitle='',
                    title1='', title2='', title3='', title4=''):
    # suptitle, xs1, ys1, zs1, xs2, ys2, zs2,
    # values3, values4, title1, title2, title3, title4
    fig = plt.figure(num=window_title, figsize=(16,9))
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.95,
                        bottom=0.07,
                        left=0.07,
                        right=0.9,
                        hspace=0.13,
                        wspace=0.15)

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.set_title(title1)
    ax2.set_title(title2)
    ax3.set_title(title3)
    ax4.set_title(title4)

    ax1.set_xlabel('PC1')
    ax2.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax2.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax2.set_zlabel('PC3')

    ax1.scatter(xs1, ys1, zs1,
               depthshade=False,
               edgecolors='C0',
               facecolors='none',
               s=40,
               label='{} elements'.format(xs1.shape[0]))
    ax2.scatter(xs2, ys2, zs2,
                depthshade=False,
                edgecolors='C0',
                facecolors='none',
                s=40,
                label='{} elements'.format(xs2.shape[0]))
    ax3.plot(values3)
    ax4.plot(values4)

    ax1.legend()
    ax2.legend()

    plt.show(block=False)
    plt.pause(0.001)
    axes = [ax1, ax2, ax3, ax4]
    return axes

# #############################################################################
# #############################################################################
def load_data_from_mat_files(patient_list, scid_list, pwd):
    # Function to load waves from .mat files patient- and scid-wise
    # patient_list := list of ints containing the patients' numbers
    # scid_list := list of strings, i.e. 'nsc', 'scp'
    # pwd := path to the folder with .mat files
    import scipy.io
    for patient_nr in patient_list:
        for scid in scid_list:
            print('Patient {} {}'.format(patient_nr, scid))
            pwd_mat = pwd + 'nfpat' + str(patient_nr) + '.icp.' + scid + 'seg.waves.mat'

            if scid == scid_list[0] and patient_nr == patient_list[0]:
                if patient_nr == 7:
                    pwd_mat1 = pwd + 'nfpat' + str(patient_nr) + '.icp.' + scid + 'seg.waves.mat'
                    pwd_mat2 = pwd + 'nfpat' + str(patient_nr) + '.testicp.' + scid + 'seg.waves.mat'
                    concatenated_waves = scipy.io.loadmat(pwd_mat1)['waves_mat']
                    concatenated_waves = np.concatenate((concatenated_waves,
                                                         scipy.io.loadmat(pwd_mat2)
                                                         ['waves_mat']),
                                                         axis = 0)
                else:
                    concatenated_waves = scipy.io.loadmat(pwd_mat)['waves_mat']
            elif patient_nr == 7:
                pwd_mat1 = pwd + 'nfpat' + str(patient_nr) + '.icp.' + scid + 'seg.waves.mat'
                pwd_mat2 = pwd + 'nfpat' + str(patient_nr) + '.testicp.' + scid + 'seg.waves.mat'
                concatenated_waves = np.concatenate((concatenated_waves,
                                                     scipy.io.loadmat(pwd_mat2)
                                                     ['waves_mat']),
                                                     axis = 0)
                concatenated_waves = np.concatenate((concatenated_waves,
                                                     scipy.io.loadmat(pwd_mat1)
                                                     ['waves_mat']),
                                                     axis = 0)
            else:
                concatenated_waves = np.concatenate((concatenated_waves,
                                                     scipy.io.loadmat(pwd_mat)
                                                     ['waves_mat']))
    return concatenated_waves

# #############################################################################
# #############################################################################
def acces_waves_by_type(index_list=None, stacked_waves=None, wave_type=None):
    # Returns the waves with index==index_list of type==wave_type
    if wave_type == 'abp':
        wave_type = 1
    elif wave_type == 'icp':
        wave_type = 0
    else:
        raise NameError('Wrong wave type!')

    if index_list == None:
        return stacked_waves[wave_type, :]
    else:
        if wave_type == 'icp':
            return stacked_waves[0, index_list, :]

        else:
            return stacked_waves[1, index_list, :]

# #############################################################################
# #############################################################################
def cut_by_size_in3_classes(a_waves, b_waves, i_waves, wave_size):
    # Cut waves from raw a- and b_waves pulses of wave_size and label them
    nr_a = len(a_waves)//wave_size
    nr_b = len(b_waves)//wave_size
    nr_i = len(i_waves)//wave_size

    nr_waves = nr_a + nr_b + nr_i

    waves = np.zeros((nr_waves, wave_size))
    labels = np.zeros((nr_waves,), dtype=int)

    for i in range(nr_waves):
        start = i*wave_size
        stop = start + wave_size

        if i < nr_a:
            #print('a waves')
            labels[i] = 0
            for j in range(start,stop):
                index = j - start
                waves[i][index] = a_waves[j]
        elif i > nr_a:
            if i < (nr_a + nr_b):

        #elif (i > nr_a) and (i < (nr_a + nr_b)):
                start = wave_size*(i - nr_a)
                stop = start + wave_size
                labels[i] = 1
                for j in range(start,stop):
                    index = j - start
                    waves[i][index] = b_waves[j]

            else:
                #print('i waves')
                start = wave_size*(i - nr_a - nr_b)
                stop = start + wave_size
                labels[i] = 2
                for j in range(start,stop):
                    #print('index:')
                    #print(index)
                    index = j - start
                    #print(index)
                    waves[i][index] = i_waves[j]

    return waves, labels

def create_dataset(a_waves=None, b_waves=None, i_waves=None, wave_size=None, use_smote=False):
    # Sweeps through raw a_waves, b_waves, cuts pulses of size=wave_size
    # Returns waves vector of shape ((len(a_waves)+len(b_waves))/wave_size, wave_size)
    # and a labels vector of shape  ((len(a_waves)+len(b_waves))/wave_size, )

    nr_a = len(a_waves)//wave_size
    nr_b = len(b_waves)//wave_size

    if i_waves.any() == False:
        nr_i = 0
    else:
        nr_i = len(i_waves)//wave_size
        print('i_present')

    nr_waves = nr_a + nr_b + nr_i
    #print('Nr of waves: {}'.format(nr_waves))

    waves = np.zeros((nr_waves, wave_size))
    labels = np.zeros((nr_waves,), dtype=int)

    # Cutting function
    if i_waves.any() == False:
        waves, labels = cut_by_size_in2_classes(a_waves, b_waves, wave_size)
    else:
        waves, labels = cut_by_size_in3_classes(a_waves, b_waves, i_waves, wave_size)

    # Oversampling, IF classes are imbalanced
    if use_smote==True:
        from imblearn.over_sampling import SMOTE
        # Perform minority oversampling using SMOTE
        sm = SMOTE(sampling_strategy='auto',random_state=2) # state was 1
        waves, labels = sm.fit_resample(waves, labels)

    assert waves.shape[0] == nr_waves
    assert waves.shape[0] == labels.shape[0]

    return waves, labels


# Function used to plot 9 images in a 3x3 grid, and writing the true and
# predicted classes below each image
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

"""
# Examples:
# Get the first images from the test-set.
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)
"""


def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = weights#session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


class PeriodicPlotter:
  def __init__(self, sec, xlabel='', ylabel='', scale=None):
    from IPython import display as ipythondisplay
    import matplotlib.pyplot as plt
    import time

    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale

    self.tic = time.time()

  def plot(self, data):
    if time.time() - self.tic > self.sec:
      plt.cla()

      if self.scale is None:
        plt.plot(data)
      elif self.scale == 'semilogx':
        plt.semilogx(data)
      elif self.scale == 'semilogy':
        plt.semilogy(data)
      elif self.scale == 'loglog':
        plt.loglog(data)
      else:
        raise ValueError("unrecognized parameter scale {}".format(self.scale))

      plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
      ipythondisplay.clear_output(wait=True)
      ipythondisplay.display(plt.gcf())

      self.tic = time.time()
