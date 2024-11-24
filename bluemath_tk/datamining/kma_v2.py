'''

- Project: Bluemath{toolkit}.datamining
- File: kma.py
- Description: KMeans algorithm
- Author: GeoOcean Research Group, Universidad de Cantabria
- Created Date: 23 January 2024
- License: MIT
- Repository: https://gitlab.com/geoocean/bluemath/toolkit/

'''

import numpy as np

import matplotlib.pyplot as plt
import itertools

#Kmeans algorithm
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap


class KMA:

    '''
    This class implements the KMeans Algorithm (KMA)
    KMA is a clustering algorithm that divides a dataset into K distinct groups based on data similarity. It iteratively assigns data points to the nearest cluster center and updates centroids until convergence, aiming to minimize the sum of squared distances. While efficient and widely used, KMA requires specifying the number of clusters and is sensitive to initial centroid selection.
    
    '''

    def __init__(self):
        self.data = []
        self.ix_scalar = []
        self.ix_directional = []
        self.minis = []
        self.maxis = []
        self.data_norm = []
        self.centroids_norm = []
        self.centroids = []
    
    def normalize(self, data):
        '''
        Normalize data subset - norm = (val - min) / (max - min)
    
        Returns:
        - data_norm: Normalized data
        '''
    
        data_norm = np.zeros(data.shape) * np.nan
    
        # Calculate maxs and mins 
        if self.minis == [] or self.maxis == []:
    
            # Scalar data
            for ix in self.ix_scalar:
                v = self.data[:, ix]
                mi = np.amin(v)
                ma = np.amax(v)
                data_norm[:, ix] = (v - mi) / (ma - mi)
                self.minis.append(mi)
                self.maxis.append(ma)
    
            self.minis = np.array(self.minis)
            self.maxis = np.array(self.maxis)
    
        # Max and mins given
        else:
    
            # Scalar data
            for c, ix in enumerate(self.ix_scalar):
                v = self.data[:, ix]
                mi = self.minis[c]
                ma = self.maxis[c]
                data_norm[:, ix] = (v - mi) / (ma - mi)
    
        # Directional data
        for ix in self.ix_directional:
            v = self.data[:, ix]
            data_norm[:, ix] = (v * np.pi / 180.0) / np.pi
    
        self.minis = self.minis
        self.maxis = self.maxis
    
        return data_norm
    
    def denormalize(self, data_norm):
        '''
        DeNormalize data
    
        Returns:
        - data: De-normalized data
        '''
    
        data = np.zeros(data_norm.shape) * np.nan
    
        # Scalar data
        for c, ix in enumerate(self.ix_scalar):
            v = data_norm[:, ix]
            mi = self.minis[c]
            ma = self.maxis[c]
            data[:, ix] = v * (ma - mi) + mi
    
        # Directional data
        for ix in self.ix_directional:
            v = data_norm[:, ix]
            data[:, ix] = v * 180  # Convert back to degrees if needed (commented out np.pi conversion)
    
        return data
    
    def kma(self, n_clusters):
        '''
        Normalize data and calculate centers using
        kmeans algorithm
    
        Args:
        - num_centers: Number of centers to calculate
        
        Returns:
        - centroids: Calculated centroids
        '''
    
        print('\nkma parameters: {0} --> {1}\n'.format(self.data.shape[0], n_clusters))


        if not np.shape(self.data)[1] == len(self.ix_scalar) + len(self.ix_directional):
            raise KMAError("ix_scalar and ix_directional should match with the number of data columns")


        data = self.data
        data_norm = self.normalize(data)
        self.data_norm = data_norm

        kma = KMeans(n_clusters=n_clusters, n_init=100).fit(data_norm)

        bmus = kma.labels_
        centroids_norm = kma.cluster_centers_
    
        # De-normalize scalar and directional data

        self.bmus = bmus
        self.centroids_norm = centroids_norm
        self.centroids = self.denormalize(self.centroids_norm)

    ### Plotting ###

    default_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    def scatter_data(self, var_names = [], norm = False, plot_centroids = False, custom_params = None):
        
        """
        Create scatter plots for all combinations of variables in the data.
        """

        scatter_defaults = {
            'figsize': (9, 8),
            'marker': '.',
            'color_data': '#00CC96',
            'color_subset': '#AB63FA',
            'alpha_data': 0.5,
            'alpha_subset': 0.7,
            'size_data': 10,
            'size_centroid': 70,
            'fontsize' : 12,
        }

        scatter_params = {**scatter_defaults, **custom_params} if custom_params else scatter_defaults


        if norm:
            data = self.data_norm
            centroids = self.centroids_norm
        else:
            data = self.data
            centroids = self.centroids
        bmus = self.bmus
        
        num_variables = data.shape[1]
        fig, axes = plt.subplots(nrows=num_variables-1, ncols=num_variables-1, figsize=scatter_params['figsize'])

        # Create scatter plots
        for i, j in itertools.combinations(range(num_variables), 2):
            
            x_data = data[:, j]
            y_data = data[:, i]

            if num_variables>2:
                ax = axes[i, j-1]
            else:
                ax = axes
            

            if plot_centroids:

                # Create scatter plots
                x_centroid = centroids[:, j]
                y_centroid = centroids[:, i]

                cmap_continuous = plt.cm.rainbow
                cmap_discretized = ListedColormap(cmap_continuous(np.linspace(0, 1, len(np.unique(bmus)))))

                im = ax.scatter(x_data, y_data, c = bmus, s = scatter_params['size_data'], 
                                     label = 'bmus', cmap = cmap_discretized, alpha = scatter_params['alpha_subset'],)

                ax.scatter(x_centroid, y_centroid, s = scatter_params['size_centroid'], 
                                     c = np.array(range(len(np.unique(bmus)))) + 1, cmap = cmap_discretized, ec = 'k',
                                     label = 'Centroids')

                plt.colorbar(im, ticks=np.arange(0, len(np.unique(bmus))))

            else:
                ax.scatter(x_data, y_data, s = scatter_params['size_data'], 
                                 c = scatter_params['color_data'], 
                                 alpha = scatter_params['alpha_data'],label = 'Data')
        
            if len(var_names)>0:
                ax.set_xlabel(var_names[j], fontsize = scatter_params['fontsize'])
                ax.set_ylabel(var_names[i], fontsize = scatter_params['fontsize'])
            else:
                ax.set_xlabel(f'x{j}', fontsize = scatter_params['fontsize'])
                ax.set_ylabel(f'x{i}', fontsize = scatter_params['fontsize'])

        for i in range(num_variables-1):
            for j in range(num_variables-1):

                if num_variables>2:
                    ax = axes[i, j]
                else:
                    ax = axes
                if i > j:
                    ax.axis('off')
                    ax.get_xaxis().set_visible(False) 
                    ax.get_yaxis().set_visible(False) 
                else:
                    ax.legend(fontsize = scatter_params['size_data'])
                    ax.tick_params(axis='both', labelsize=scatter_params['fontsize']) 


        plt.tight_layout()
        plt.show()


    def scatter_subset(self, var_names = [], norm = False, custom_params = None):

        self.scatter_data(var_names = var_names, norm = norm, plot_centroids = True, custom_params = custom_params)
        plt.show()

class KMAError(Exception):
    """Custom exception for KMA class."""
    def __init__(self, message="KMA error occurred."):
        self.message = message
        super().__init__(self.message)


