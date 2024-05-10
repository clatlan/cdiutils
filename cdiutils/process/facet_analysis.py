import os
import shutil
from os.path import dirname
import h5py
import numpy as np
from numpy.linalg import norm as npnorm
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from scipy import ndimage
from scipy.ndimage import binary_erosion as erosion
from collections import defaultdict
import json
import copy
from scipy.optimize import minimize
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from cdiutils.plot.formatting import get_figure_size

from cdiutils.utils import (rotation_x,
                            rotation_y,
                            rotation_z,
                            rotation,
                            error_metrics,
                            retrieve_original_index
)

class FacetAnalysisProcessor:
    """
    A class to bundle all functions needed to determine the surface, 
    the support, and anylyse the facets.
    """

    def __init__(
            self,
            parameters: dict,
    ) -> None:
        
        #Parameters
        self.params = parameters["cdiutils"]
        
        self.raw_process = self.params["raw_process"]
        
        self.remove_edges = self.params["remove_edges"]
        self.nb_nghbs_min = self.params["nb_nghbs_min"]
        self.nb_facets = self.params["nb_facets"]
        self.top_facet_ref_index = self.params["top_facet_reference_index"]
        self.authorized_index = self.params["authorized_index"]
        if self.params["index_to_display"] is None:
            self.index_to_display = [self.top_facet_ref_index]
        else:
            self.index_to_display = self.params["index_to_display"]
        self.size = self.params["size"]
        self.display_f_e_c = self.params["display_f_e_c"]
        self.input_parameters = None
        self.previous_input_parameters = None

        #Global variables
        
        self.support = None

        #Path

        self.order_of_derivative = self.params["order_of_derivative"]
        
        if self.nb_facets is None:
            raise ValueError("Please indicate the expected number of facets :"
                             "\"nb_facets\" = n ."
            )

        self.dump_dir = self.params["metadata"]["dump_dir"]
        if self.params["support_path"] is None:
 
            if self.params["method_det_support"]=="Amplitude_variation":
                self.path_surface = (f'{self.dump_dir}surface_calculation/'
                                     f'{self.params["method_det_support"]}/'
                )
            elif self.params["method_det_support"]=="Isosurface":
                self.path_surface = (f'{self.dump_dir}surface_calculation/'
                                     f'{self.params["method_det_support"]}'
                                     f'={self.params["isosurface"]}/'
                )
            else:
                raise ValueError("Unknown method_det_support. "
                                 "Use method_det_support='Amplitude_variation'"
                                  " or method_det_support='Isosurface' ")

            if self.params["method_det_support"]=="Amplitude_variation":
                if not self.order_of_derivative is None:
                    self.path_order = (f'{self.path_surface}'
                                       f'{self.order_of_derivative}/'
                    )
                else:
                    raise ValueError("Unknown order_of_derivative. "
                                     "Use order_of_derivative = 'Gradient' "
                                     "or order_of_derivative = 'Laplacian'")
            elif self.params["method_det_support"]=="Isosurface":
                self.path_order = self.path_surface
            
            self.path_f = (f'{self.path_order}nb_facets='
                                     f'{self.nb_facets}/'
            )

            h5_file_path = (f'{self.dump_dir}/cdiutils_S'
                            f'{self.params["metadata"]["scan"]}'
                            f'_structural_properties.h5'
            )

            with h5py.File(h5_file_path, 'r') as file:
            # Check if 'volume' dataset exists in the file
                if 'volumes' in file:
                    volumes_dataset = file['volumes']
                    if 'support' in volumes_dataset:
                        self.support = np.array(volumes_dataset['support'])
                        print('support was loaded successfully.', '\n')
                    else:
                        raise FileNotFoundError('Error: The dataset "support"'
                                                'does not exist '
                                                'in the "volume" dataset.'
                        )
                else:
                    raise FileNotFoundError(f'Error:The dataset "support" '
                          f'does not exist in the file {h5_file_path}.'
                    )

        else : 
            self.path_f = (f'{dirname(self.params["support_path"])}/'
                                     f'nb_facets={self.nb_facets}/'
            )
            self.support=np.load(self.params["support_path"])

        
        self.path_visu = f'{self.path_f}/visualization/'
        
        self.X, self.Y, self.Z = np.shape(self.support)
        self.surface = (self.support 
                        - erosion(self.support).astype(int)
        )


    def check_previous_data(self) -> None:
        """
        If one of these parameters 
        (authorized_index, top_facet_reference_index,
        remove_edges or nb_nghbs_min) 
        has changed since the previous run, 
        this method deletes the files affected.
        """

        if self.authorized_index is None:
            raise ValueError("Please indicate the authorised index. \n"
                             "\"self.authorized_index\" = ['max', n] "
                             "if you want |h|, |k|, |l| <= n. \n"
                             "\"self.authorized_index\" ="
                             "['absolute', a,b, ...] "
                             "if you want |h|, |k|, |l| in [a,b, ...]. \n"
                             "\"self.authorized_index\" = "
                             "['families', [1,0,0], [1,1,1], ...] "
                             "if you want to authorise the index families "
                             "[1,0,0], [1,1,1], ..."
            )
       
        if self.top_facet_ref_index is None:
            raise ValueError("Please indicate the reference index of "
                             "the upper facet : \"top_facet_reference_index\""
                             " = [h,k,l] ."
            )

        self.input_parameters=[self.authorized_index, 
                               self.top_facet_ref_index,
                               self.remove_edges,
                               self.nb_nghbs_min,
                               self.index_to_display,
                               self.size
        ]
        
        try: 
            with open(f'{self.path_f}input_parameters.json', 'r') as f:
                self.previous_input_parameters = json.load(f)

            if self.input_parameters[2] != self.previous_input_parameters[2]:
                print('remove_edges has been changed')
                try:
                    shutil.rmtree(self.path_f)
                    print(f'The folder at {self.path_f}'
                          f' has been successfully removed.'
                    )
                except:
                    print(f'The folder {self.path_f} doesn\'t exist.')
          
            elif self.input_parameters[3] != self.previous_input_parameters[3]:
                print('nb_nghbs_min has been changed')
                files_to_remove = ['facet_label.npy', 
                                   'edge_label.npy',
                                   'corner_label.npy', 
                                   'nb_edges_corners.npy',
                                   'smooth_dir_facet.npy', 
                                   'smooth_dir_edge.npy',
                                   'smooth_dir_corner.npy', 
                                   'facet_label_index.json',
                                   'edge_label_index.json', 
                                   'corner_label_index.json',
                                   'rotation_matrix.npy'
                ]
                for file in files_to_remove:
                    try:
                        os.remove(f'{self.path_f}/{file}')
                        print(f'The file {file} '
                              f'has been successfully removed.')
                    except OSError as e:
                        print(f'The file {file} doesn\'t exist.')
                try:
                    shutil.rmtree(self.path_visu)
                    print(f'The folder at {self.visualization}'
                          f' has been successfully removed.'
                    )
                except:
                    print(f'The folder {self.path_visu}'
                          f'doesn\'t exist.'
                    )

            elif ((self.input_parameters[0][0] != 
                  self.previous_input_parameters[0][0]
                  )
                or not np.array_equal(self.input_parameters[0][1:], 
                                   self.previous_input_parameters[0][1:]
                    ) 
                or not np.array_equal(self.input_parameters[1], 
                                    self.previous_input_parameters[1]
                        )
            ):
                print('authorized_index or top_facet_reference_index'
                      'has been changed'
                )
                files_to_remove = ['facet_label_index.json',
                                   'edge_label_index.json', 
                                   'corner_label_index.json',
                                   'rotation_matrix.npy'
                ]
                for file in files_to_remove:
                    try:
                        os.remove(f'{self.path_f}/{file}')
                        print(f'The file {file} '
                              f'has been successfully removed.')
                    except OSError as e:
                        print(f'The file {file} doesn\'t exist.')
                try:
                    shutil.rmtree(self.path_visu)
                    print(f'The folder at {self.visualization}'
                          f' has been successfully removed.'
                    )
                except:
                    print(f'The folder {self.path_visu} doesn\'t exist.')

            elif (self.input_parameters[5] != self.previous_input_parameters[5]
                  or not np.array_equal(self.input_parameters[4], 
                                       self.previous_input_parameters[4]
                          )
                 ):
                print('index_to_display or size has been changed')
                try:
                    shutil.rmtree(self.path_visu)
                    print(f'The folder at {self.visualization}'
                          f' has been successfully removed.'
                    )
                except:
                    print(f'The folder {self.path_visu} doesn\'t exist.')
        except:
            print("No previous input_parameters found")
            
        os.makedirs(self.path_f, exist_ok=True)
        
        with open(f'{self.path_f}input_parameters.json', 'w') as f:
                json.dump(list(self.input_parameters), f, indent=2)
        
        os.makedirs(self.path_visu, exist_ok=True)


### Facet analysis

    def def_smooth_supp_dir(self) -> None: 
        """
        Determine a smooth surface normal direction 
        at each surface voxel from the given support.
        """
        try:
            smooth_dir=np.load(f'{self.path_f}/smooth_supp_dir.npy')

        except:
            print('No previous smooth_gradient or smooth_supp_dir found')
            smooth_dir = np.zeros((self.X, self.Y, self.Z, 3))
            first_erosion = erosion(self.support).astype(self.support.dtype)
            second_erosion = erosion(first_erosion).astype(first_erosion.dtype)
            mountain_support = self.support + first_erosion + second_erosion
            gradient = np.array(np.gradient(mountain_support))
            gradient = np.moveaxis(gradient, 0, 3)
            gradient = np.array(gradient * self.surface[:,:,:,None])

            for x, y, z in zip(*np.nonzero(self.surface)):
                voxel=np.array([x,y,z])
                smooth_grad=np.array([0,0,0])
                nb_voisins = 0
                # Define 27 possible offsets
                offsets = [(dx, dy, dz) 
                           for dx in [-1, 0, 1] 
                           for dy in [-1, 0, 1] 
                           for dz in [-1, 0, 1]
                ]

                for offset in offsets:
                    adj_voxel = tuple(x + y for x, y in zip(voxel, offset))
                    if self.surface[adj_voxel]>0:
                        grad_adj_voxel=gradient[adj_voxel]
                        smooth_grad = smooth_grad + grad_adj_voxel
                        nb_voisins += 1
                smooth_grad /= nb_voisins
                smooth_dir[x,y,z] = (-smooth_grad
                                          /(npnorm(smooth_grad) 
                                            if npnorm(smooth_grad) !=0 
                                            else 1)
                )
                
            np.save(f'{self.path_f}/smooth_supp_dir.npy', smooth_dir)
            
        return smooth_dir


    def find_edges(self, smooth_dir) -> None: 
        
        try:
            edges=np.load(f'{self.path_f}/edges.npy')
            
        except:
            print('No previous edges found')
            edges = np.zeros((self.X, self.Y, self.Z))
            n=2
            for x, y, z in zip(*np.nonzero(self.surface > 0)):
                if n<x<self.X-n and n<y<self.Y-n and n<z<self.Z-n:

                    directions= smooth_dir[x-n:x+n+1,y-n:y+n+1,z-n:z+n+1]
                    dir_center = directions[n,n,n]
                    
                    vecteurs = [directions[i,j,k] 
                                for i in range(2*n+1) 
                                for j in range(2*n+1) 
                                for k in range(2*n+1) 
                                if (np.any(directions[i,j,k] != dir_center) 
                                    and np.any(directions[i,j,k] != [0,0,0])
                                )
                    ]
        
                    if len(vecteurs) > 0:

                        eps = 0.25 
                        min_samples = 5 

                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

                        clusters = dbscan.fit_predict(vecteurs)

                        n_clusters_ = len(set(clusters))
                        n_noise_ = list(clusters).count(-1)

                        if n_clusters_ > 1: 
                            edges[x,y,z]=1
                            
            np.save(f'{self.path_f}/edges.npy', edges)
            
        return edges
 
    def def_list_facet_analysis(self, smooth_dir, edges = None) -> None:
        try:
            cluster_list = np.load(f'{self.path_f}/cluster_list.npy')
            cluster_pos = np.load(f'{self.path_f}/cluster_pos.npy')
        except FileNotFoundError:
            print('No previous cluster_list or cluster_pos found')

            cluster_list = []
            cluster_pos = np.zeros((self.X, self.Y, self.Z))
            n = 0
            if self.remove_edges:
                surface_to_cluster = self.surface*(1-edges)
            else:
                surface_to_cluster = self.surface
            for i, j, k in zip(*np.nonzero(surface_to_cluster)):
                direction = smooth_dir[i][j][k]
                cluster_list.append(direction)
                cluster_pos[i][j][k] = n
                n += 1
                
            np.save(f'{self.path_f}/cluster_list.npy', cluster_list)
            np.save(f'{self.path_f}/cluster_pos.npy', cluster_pos)
        
        return cluster_list, cluster_pos

          
    def def_label(self, cluster_list, cluster_pos, edges = None) -> None:
        try:
            label = np.load(f'{self.path_f}/label.npy')
        except:
            print('No previous label found')

            clustering = SpectralClustering(n_clusters=self.nb_facets, 
                                            assign_labels='kmeans', 
                                            random_state=0
            )
            clustering = clustering.fit(cluster_list)
            spect_labels = clustering.labels_
 
        
            label = np.zeros((self.X, self.Y, self.Z))
            if self.remove_edges:
                surface_to_cluster = self.surface*(1-edges)
            else:
                surface_to_cluster = self.surface
            for i, j, k in zip(*np.nonzero(surface_to_cluster)):
                label[i, j, k] = spect_labels[int(cluster_pos[i, j, k])]
                label[i, j, k] = int(label[i, j, k] +1)
                    
            np.save(f'{self.path_f}/label.npy', label)
            
        return label
    
    def distance_clustering(self, pos_a, pos_b, dir_a, dir_b):
        return npnorm(pos_b-pos_a) * npnorm(dir_b-dir_a) 

    def clustering(self, smooth_dir, label) -> None:
        try:
            facet_label=np.load(f'{self.path_f}/facet_label.npy')
        except:
            print('No previous facet_label found')
    
        facet_label = np.zeros((self.X,self.Y,self.Z))
        pos_centres=[]
        dir_centres=[]
        for lbl in range(1, self.nb_facets+1):
            pos_centre= np.array([0,0,0])
            dir_centre= np.array([0,0,0])
            n_points=0
            for x,y,z in zip(*np.nonzero(label == lbl)):
                pos_centre = pos_centre + np.array([x,y,z])
                dir_centre = dir_centre + smooth_dir[x,y,z]
                n_points += 1
            pos_centre = pos_centre/n_points
            dir_centre = dir_centre/n_points
            pos_centres.append(pos_centre)
            dir_centres.append(dir_centre)

        for x,y,z in zip(*np.nonzero(self.surface>0)):
            lbl=min([self.distance_clustering(np.array([x,y,z]), 
                                                pos_centres[i], 
                                                smooth_dir[x,y,z], 
                                                dir_centres[i]), 
                        i] 
                        for i in range(len(pos_centres))
                       )[1]+1
            facet_label[x,y,z]=lbl

        if self.nb_nghbs_min > 0:
            facet_label = np.copy(facet_label)
            for x,y,z in zip(*np.nonzero(self.surface>0)):
                lbl = facet_label[x,y,z]
                nb_voisins = 0
                nghbs_label={}
                # Define 27 possible offsets
                offsets = [(dx, dy, dz) 
                            for dx in [-1, 0, 1] 
                            for dy in [-1, 0, 1] 
                            for dz in [-1, 0, 1]
                ]
                for offset in offsets:
                    adj_voxel = tuple(x + y for x, y 
                                      in zip(np.array([x,y,z]), offset)
                    )
                    label_adj_voxel = facet_label[adj_voxel]
                
                    if label_adj_voxel == lbl:
                        nb_voisins += 1
                    if label_adj_voxel > 0:
                        if not label_adj_voxel in nghbs_label:
                            nghbs_label[label_adj_voxel] = 1
                        else:
                            nghbs_label[label_adj_voxel] += 1
                            
                if nb_voisins < self.nb_nghbs_min:
                    new = max(nghbs_label, key = lambda k: nghbs_label[k])
                    facet_label[x,y,z] = new
        
        np.save(f'{self.path_f}/facet_label.npy', facet_label)
        
        return facet_label


    def def_edge_corner(self, facet_label) -> None:
        try: 
            edge_label=np.load(f'{self.path_f}/edge_label.npy')
            corner_label=np.load(f'{self.path_f}/corner_label.npy')
            nb_edges, nb_corners = np.load(f'{self.path_f}/'
                                                     f'nb_edges_corners.npy'
            )
            print('nb_edges = ', nb_edges)
            print('nb_corners = ', nb_corners)
        
        except:
            print('No previous edge_label or corner_label found')
            
            edge_label = copy.deepcopy(facet_label)
            corner_label = copy.deepcopy(facet_label)
            
            corner_lbl = self.nb_facets + 1
            edge_lbl = self.nb_facets + 2
            
            for i,j,k in zip(*np.nonzero(facet_label>0)):
                
                voxel=[i,j,k]

                dict_label={}
                # Define 27 possible offsets
                offsets = [(dx, dy, dz) 
                           for dx in [-1, 0, 1] 
                           for dy in [-1, 0, 1] 
                           for dz in [-1, 0, 1]
                ]
                for offset in offsets:
                    adj_voxel = tuple(x + y for x, y in zip(voxel, offset))

                    if (0 <= adj_voxel[0] < self.X 
                        and 0 <= adj_voxel[1] < self.Y 
                        and 0 <= adj_voxel[2] < self.Z
                    ):
                        label_adj_voxel=facet_label[adj_voxel]
                        if label_adj_voxel >= 1:
                            if not label_adj_voxel in dict_label:
                                dict_label[label_adj_voxel] = 1
                            else:
                                dict_label[label_adj_voxel] += 1

                if len(dict_label)>=2:
                    edge_label[i][j][k] = edge_lbl

                if len(dict_label)==2:
                    corner_label[i][j][k] = edge_lbl
                elif len(dict_label)>2:
                    corner_label[i][j][k] = corner_lbl

            labels_edges={}
            lbl = edge_lbl + 1
            for i, j, k in zip(*np.nonzero(edge_label == edge_lbl)):
                nghbs_label={}
                # Define 27 possible offsets
                offsets = [(dx, dy, dz) 
                           for dx in [-1, 0, 1] 
                           for dy in [-1, 0, 1] 
                           for dz in [-1 , 0, 1]
                ]
                for offset in offsets:
                    adj_voxel = tuple(x + y for x, y 
                                      in zip(np.array([i,j,k]), offset)
                    )
                    label_adj_voxel = facet_label[adj_voxel]
                    if label_adj_voxel > 0:
                        if not label_adj_voxel in nghbs_label:
                            nghbs_label[label_adj_voxel] = 1
                        else:
                            nghbs_label[label_adj_voxel] += 1
 
                label_1 = max(nghbs_label, key = lambda k: nghbs_label[k])
                nghbs_label[label_1]=0
                label_2 = max(nghbs_label, key = lambda k: nghbs_label[k])
                labels = list([label_1, label_2])
                labels.sort()
                labels = tuple(labels)
                if not labels in labels_edges: 
                    edge_label[i,j,k] = lbl
                    if corner_label[i,j,k] == edge_lbl:
                        corner_label[i,j,k] = lbl
                    labels_edges[labels] = lbl 
                    lbl += 1
                else :
                    edge_label[i,j,k] = labels_edges[labels]
                    if corner_label[i,j,k] == edge_lbl:
                        corner_label[i,j,k] = labels_edges[labels]
            
            nb_edges = len(labels_edges.keys())
            print('nb_edges = ', nb_edges)
         
            list_corner=[]
            list_corner_pos=np.zeros((self.X,self.Y,self.Z))
            n=0
            for x, y, z in zip(*np.nonzero(corner_label==corner_lbl)):
                list_corner.append([x,y,z])
                list_corner_pos[x,y,z]=n
                n += 1

            eps = 1.74 
            min_samples = 1 

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)

            clusters = dbscan.fit(list_corner)
            spect_labels = clusters.labels_

            nb_corners = len(np.unique(spect_labels))
            print('nb_corners = ', nb_corners)
            for i, j, k in zip(*np.nonzero(corner_label==corner_lbl)):
                pos_label = int(list_corner_pos[i, j, k])
                spect_label_corner = spect_labels[pos_label] 
                corner_label[i, j, k] = int( spect_label_corner
                                                 + self.nb_facets + 3 
                                                 + nb_edges + 2
                )

            np.save(f'{self.path_f}/edge_label.npy', edge_label)
            np.save(f'{self.path_f}/corner_label.npy', corner_label)
            np.save(f'{self.path_f}/nb_edges_corners.npy', 
                    np.array([nb_edges, nb_corners])
            )
        
        return edge_label, corner_label, nb_edges
 

    def def_smooth_dir_facet_edge_corner(self, smooth_dir, facet_label, 
                                         edge_label, corner_label
    ) -> None:
        
        try:
            smooth_dir_f = np.load(f'{self.path_f}/'
                                        f'smooth_dir_facet.npy'
            )
            smooth_dir_e = np.load(f'{self.path_f}/'
                                        f'smooth_dir_edge.npy'
            )
            smooth_dir_c = np.load(f'{self.path_f}/'
                                        f'smooth_dir_corner.npy'
            )

        except:
            print('No previous smooth_dir_facet, '
                   'smooth_dir_edge or smooth_dir_corner found'
            )
            smooth_dir_f = np.zeros((int(np.max(facet_label))+1,3))
            nb_directions_facet = defaultdict(int)
            
            smooth_dir_e = np.zeros((int(np.max(edge_label))+1,3))
            nb_directions_edge = defaultdict(int)
            
            smooth_dir_c = np.zeros((int(np.max(corner_label))+1,3))
            nb_directions_corner = defaultdict(int)

            for i, j, k in zip(*np.nonzero(self.surface > 0)):
                facet_lbl = int(facet_label[i, j, k])
                edge_lbl = int(edge_label[i, j, k])
                corner_lbl = int(corner_label[i, j, k])
                
                if facet_lbl > 0:
                    smooth_dir_f[facet_lbl] += smooth_dir[i, j, k]
                    nb_directions_facet[facet_lbl] += 1

                if edge_lbl > 0:
                    smooth_dir_e[edge_lbl] += smooth_dir[i, j, k]
                    nb_directions_edge[edge_lbl] += 1
                    
                if corner_lbl > 0:
                    smooth_dir_c[corner_lbl] += smooth_dir[i, j, k]
                    nb_directions_corner[corner_lbl] += 1

            for lbl in range(1, int(np.max(corner_label))+1):
                if lbl in nb_directions_facet:
                    smooth_dir_f[lbl] /= nb_directions_facet[lbl]
                    smooth_dir_f[lbl] /= npnorm(smooth_dir_f[lbl])
                    
                if lbl in nb_directions_edge:
                    smooth_dir_e[lbl] /= nb_directions_edge[lbl]
                    smooth_dir_e[lbl] /= npnorm(smooth_dir_e[lbl])
                    
                if lbl in nb_directions_corner:
                    smooth_dir_c[lbl] /= nb_directions_corner[lbl]
                    smooth_dir_c[lbl] /= npnorm(smooth_dir_c[lbl])

            np.save(f'{self.path_f}/smooth_dir_facet.npy',
                     smooth_dir_f
            )
            np.save(f'{self.path_f}/smooth_dir_edge.npy', 
                    smooth_dir_e
            )
            np.save(f'{self.path_f}/smooth_dir_corner.npy', 
                    smooth_dir_c
            )
        
        return smooth_dir_f, smooth_dir_e, smooth_dir_c


    def def_index(self, smooth_dir_f, smooth_dir_e, smooth_dir_c, facet_label, 
                  edge_label, corner_label, nb_edges
    ) -> None:

        facet_directions = []
        weights = {}
        total_weight = np.sum(facet_label>0)
        for lbl in list(np.unique(facet_label)):
            if lbl>0:
                facet_directions.append([lbl, smooth_dir_f[int(lbl)]])
                mask = (facet_label == lbl)
                weights[lbl] = np.sum(mask)/total_weight

        ## Determine the top facet 
        
        direction_top_facet=[0,0,0]
        for direction in facet_directions:
            lbl = direction[0]
            if direction[1][1]>direction_top_facet[1]:
                direction_top_facet=direction[1] 
                
        ## Calculate the coordinates of the normalized reference_index 
        ref_dir_top_facet = (self.top_facet_ref_index
                             / npnorm(self.top_facet_ref_index)
        )
    
        ## Create the authorized coordinates set 

        if self.authorized_index[0] == 'max':
            authorized_h = np.linspace(-self.authorized_index[1], 
                                       self.authorized_index[1], 
                                       2*self.authorized_index[1]+1
            )
            authorized_k = np.linspace(-self.authorized_index[1], 
                                       self.authorized_index[1], 
                                       2*self.authorized_index[1]+1
            )
            authorized_l = np.linspace(-self.authorized_index[1], 
                                       self.authorized_index[1], 
                                       2*self.authorized_index[1]+1
            )
            authorized_index = np.array(np.meshgrid(authorized_h, 
                                                    authorized_k, 
                                                    authorized_l))
            
            authorized_index = authorized_index.T.reshape(-1, 3)
 
        elif self.authorized_index[0] == 'absolute':
            authorized_h = []
            authorized_k = []
            authorized_l = []
            for i in range(1,len(self.authorized_index)):
                if self.authorized_index[i] != 0:
                    authorized_h.append(self.authorized_index[i])
                    authorized_h.append(-self.authorized_index[i])
                    authorized_k.append(self.authorized_index[i])
                    authorized_k.append(-self.authorized_index[i])
                    authorized_l.append(self.authorized_index[i])
                    authorized_l.append(-self.authorized_index[i])
                else:
                    authorized_h.append(self.authorized_index[i])
                    authorized_k.append(self.authorized_index[i])
                    authorized_l.append(self.authorized_index[i])     

            authorized_index = np.array(np.meshgrid(authorized_h, 
                                                    authorized_k, 
                                                    authorized_l))
                
            authorized_index = authorized_index.T.reshape(-1, 3)
  
        elif self.authorized_index[0] == 'families':
            authorized_index =[]
            added = {} 
            for i in range(1, len(self.authorized_index)):
                for s0 in range(-1, 2, 2):
                    for s1 in range(-1, 2, 2):
                        for s2 in range(-1, 2, 2):
                            for x in range(3):
                                for y in range(2):
                                    a_i_i = self.authorized_index[i]
                                    index_i = list(copy.deepcopy(a_i_i))
                                    index = []
                                    index.append(s0*index_i[x])
                                    index_i.pop(x)
                                    index.append(s1*index_i[y])
                                    index_i.pop(y)
                                    index.append(s2*index_i[0])
                                    if not tuple(index) in added: 
                                        added[tuple(index)] = True 
                                        authorized_index.append(index)
            
        elif self.authorized_index[0] == 'list':
            authorized_index = self.authorized_index[1:]
            
        else: 
            raise ValueError("Wrong value of authorized_index. "
                             "Use authorized_index = ['max', integer], "
                             "authorized_index = ['absolute', integers], "
                             "authorized_index = ['families', "
                             "3-length integers lists], "
                             "or authorized_index = ['list', "
                             "3-length integers lists] "
                ) 
            
        authorized_index = np.array(authorized_index)
        authorized_index = authorized_index.astype(np.float64)
        authorized_index = authorized_index[np.any(authorized_index 
                                                   != [0., 0., 0.], 
                                                   axis=1)]
        
        
        dict_index = {}
        authorized_coordinates=[]
        for i in range(len(authorized_index)):
            index = authorized_index[i]
            normalized_index = index / npnorm(index) 
            if not tuple(normalized_index) in dict_index:
                dict_index[tuple(normalized_index)] = index
                authorized_coordinates.append(normalized_index)
            else: 

                if npnorm(index) < npnorm(dict_index[tuple(normalized_index)]):
                    dict_index[tuple(normalized_index)] = index
            
        if len(authorized_coordinates)<self.nb_facets:
            raise ValueError("Please note that the number of indices (h,k,l) "
                             "allowed is strictly less than the number of "
                             "facets requested. It is therefore not possible "
                             "to index the facets of the particle with the "
                             "authorized indices given. Note that in the case "
                             "where authorized_index is a list of values that "
                             "each h,k, and l can take "
                             "(e.g. authorized_index = [0,1,3]), it is "
                             "important to specify the value 0 if you wish "
                             "to authorize indices h,k,l to be equal to 0. "
                             "The value 0 is not considered by default. Also, "
                             "be aware that some indices are equivalent "
                             "(e.g. [1,1,1] is equivalent to [2,2,2]).")

        
        ### Rotating the system to match the reference direction 
        ### on the reference index 

        ## Determine the rotation to match the reference direction 
        ## on the reference index 

        # Initial guess for the rotation angle (alpha1_x, alpha1_y, alpha1_z)
        initial_angles1 = [0, 0, 0]

        # Define the objective function to minimize
        def objective_function1(angles):
            # Extract Euler angles from the parameters
            alpha_x, alpha_y, alpha_z = angles
            new_direction_top_facet = copy.deepcopy(direction_top_facet)
            new_direction_top_facet = np.dot(rotation_x(alpha_x), 
                                             new_direction_top_facet
            )
            new_direction_top_facet = np.dot(rotation_y(alpha_y), 
                                             new_direction_top_facet
            )
            new_direction_top_facet = np.dot(rotation_z(alpha_z), 
                                             new_direction_top_facet
            )
            return error_metrics(new_direction_top_facet, ref_dir_top_facet)

        # Set bounds for the optimization
        angle_bounds1 = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

        result1 = minimize(objective_function1, initial_angles1,
                bounds=angle_bounds1)
            
        # Get the optimal angles
        optimal_angles1 = result1.x
        alpha1_x, alpha1_y, alpha1_z = optimal_angles1
    
        ## Rotating label directions to match the reference direction 
        ## on the reference index

        inter_dir_top_facet = copy.deepcopy(direction_top_facet)

        inter_dir_top_facet = np.dot(rotation_x(alpha1_x), inter_dir_top_facet)
        inter_dir_top_facet = np.dot(rotation_y(alpha1_y), inter_dir_top_facet)
        inter_dir_top_facet = np.dot(rotation_z(alpha1_z), inter_dir_top_facet)

        inter_facet_directions = copy.deepcopy(facet_directions)
        for i in range(len(inter_facet_directions)):
            label_direction = inter_facet_directions[i]
            lbl = label_direction[0]
            direction = label_direction[1]
            new_direction = copy.deepcopy(direction)
            new_direction = np.dot(rotation_x(alpha1_x), new_direction)
            new_direction = np.dot(rotation_y(alpha1_y), new_direction)
            new_direction = np.dot(rotation_z(alpha1_z), new_direction)
            inter_facet_directions[i] = [lbl, new_direction]
        

        ### Rotating the system to match each direction on a Miller index 
        ### with the constraint that the reference direction is associated 
        ### with the reference index
                    


        # Define the objective function to minimize
        def objective_function2(angle):
            auth_coord_temp = copy.deepcopy(authorized_coordinates)
            new_directions = {}
            closest_lbl_idx = {}
            closest_idx_lbl = {}
            closest_idx_lbls = {}
            queue = []
            
            for label_direction in inter_facet_directions:
                lbl, direction = label_direction
                queue.append(lbl)
                new_direction = copy.deepcopy(direction)
                rotation_matrix=rotation(inter_dir_top_facet, angle[0])
                new_direction = np.dot(rotation_matrix, new_direction)
                new_directions[lbl] = new_direction
                
                closest_index = min(auth_coord_temp, 
                                    key=lambda coords: 
                                    error_metrics(coords, new_direction)
                )

                error = error_metrics(new_direction, closest_index)
                if not lbl in closest_lbl_idx:
                    closest_lbl_idx[lbl] = [closest_index, error]
                else: 
                    if error < closest_lbl_idx[lbl][1]:
                        closest_lbl_idx[lbl] = [closest_index, error]
                if not tuple(closest_index) in closest_idx_lbl:
                    closest_idx_lbl[tuple(closest_index)] = lbl
                    closest_idx_lbls[tuple(closest_index)] = [lbl]
                else: 
                    closest_idx_lbls[tuple(closest_index)].append(lbl)
                    if error < closest_lbl_idx[lbl][1]:
                        closest_idx_lbl[tuple(closest_index)] = lbl
                        
            total_error = 0  
            n = 0
            while len(queue)>0: 
                label_to_be_treated = list(closest_idx_lbl.values())
                for lbl in label_to_be_treated:
                    index, error = closest_lbl_idx[lbl]
                    total_error += weights[lbl]*error
                    n += 1

                    auth_coord_temp = [coord for coord in auth_coord_temp 
                                       if not np.array_equal(coord, index)
                    ]
                    del closest_idx_lbl[tuple(index)]
                    del closest_lbl_idx[lbl]
                    closest_idx_lbls[tuple(index)].remove(lbl)
                    queue.remove(lbl)
                    for lbl in closest_idx_lbls[tuple(index)]:
                        closest = min(auth_coord_temp, 
                                            key=lambda coords: 
                                            error_metrics(coords, 
                                                          new_directions[lbl]
                                            )
                        )

                        error = error_metrics(new_directions[lbl], 
                                              closest
                        )
                        if (np.all(closest_lbl_idx[lbl][0] == index) 
                            or error < closest_lbl_idx[lbl][1]
                        ):
                            closest_lbl_idx[lbl] = [closest, error]

                            closest_idx_lbl[tuple(closest)] = lbl
                            if not tuple(closest) in closest_idx_lbls:
                                closest_idx_lbls[tuple(closest)] = [lbl]
                            else:
                                closest_idx_lbls[tuple(closest)].append(lbl)
                    
            total_error /= n  
            return total_error

        list_angle=[]
        list_error=[]
        angle_min=0
        error_min=objective_function2([angle_min])
        for k in range(100):
            angle = -np.pi+k*2*np.pi/100
            list_angle.append(angle)
            error = objective_function2([angle])
            list_error.append(error)
            if error<error_min:
                angle_min = angle
                error_min = error
              
        angle_min = min((angle_min)%(2*np.pi), 
                        (angle_min + 2*np.pi/3)%(2*np.pi), 
                        (angle_min + 2*2*np.pi/3)%(2*np.pi)
                    )
        
        figsize = get_figure_size()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        ax.plot(list_angle, list_error, label="Error")

        ax.set_xlabel(r"Angle")
        ax.set_ylabel("Error")
        ax.legend()
        fig.suptitle(r"Error(angle)")
        fig.tight_layout()
        plt.show()
    
    
        # Initial guess for the rotation angle
        initial_angle2 = angle_min 

        angle_bounds2 = [(-np.pi, np.pi)]

        result2 = minimize(objective_function2, initial_angle2,
                bounds=angle_bounds2)


        # Get the optimal angles
        optimal_angle2 = result2.x
        print("Total error for indexing : ", 
              objective_function2(optimal_angle2), '\n'
        )


        final_facet_directions = copy.deepcopy(inter_facet_directions)
        for i in range(len(final_facet_directions)):
            label_direction = final_facet_directions[i]
            lbl = label_direction[0]
            direction = label_direction[1]
            new_direction = copy.deepcopy(direction)
            new_direction = np.dot(rotation(inter_dir_top_facet,
                                            optimal_angle2[0]
                                    ), 
                                   new_direction
            )

            final_facet_directions[i] = [lbl, new_direction]
            
        facet_label_index = []

        auth_coord_temp = copy.deepcopy(authorized_coordinates)
        directions = {}
        closest_lbl_idx = {}
        closest_idx_lbl = {}
        closest_idx_lbls = {}
        queue = []
            
        for label_direction in final_facet_directions:
            lbl, direction = label_direction
            directions[lbl] = direction
            queue.append(lbl)

            closest_index = min(auth_coord_temp, 
                                key=lambda coords: 
                                error_metrics(coords, direction)
            )
            error = error_metrics(direction, closest_index)
            if not lbl in closest_lbl_idx:
                closest_lbl_idx[lbl] = [closest_index, error]
            else: 
                if error < closest_lbl_idx[lbl][1]:
                    closest_lbl_idx[lbl] = [closest_index, error]
            if not tuple(closest_index) in closest_idx_lbl:
                closest_idx_lbl[tuple(closest_index)] = lbl
                closest_idx_lbls[tuple(closest_index)] = [lbl]
            else: 
                closest_idx_lbls[tuple(closest_index)].append(lbl)
                if error < closest_lbl_idx[lbl][1]:
                    closest_idx_lbl[tuple(closest_index)] = lbl
                        
        while len(queue)>0: 
            label_to_be_treated = list(closest_idx_lbl.values())
            for lbl in label_to_be_treated:
                index, error = closest_lbl_idx[lbl]
                facet_label_index.append([lbl, index])
                auth_coord_temp = [coord for coord in auth_coord_temp 
                                   if not np.array_equal(coord, index)
                ]
                del closest_idx_lbl[tuple(index)]
                del closest_lbl_idx[lbl]
                closest_idx_lbls[tuple(index)].remove(lbl)
                queue.remove(lbl)
                for lbl in closest_idx_lbls[tuple(index)]:
                    closest_index = min(auth_coord_temp, 
                                        key=lambda coords: 
                                        error_metrics(coords, 
                                                      directions[lbl]
                                        )
                    )

                    error = error_metrics(directions[lbl], closest_index)
                    if (np.all(closest_lbl_idx[lbl][0] == index) 
                        or error < closest_lbl_idx[lbl][1]
                    ):
                        closest_lbl_idx[lbl] = [closest_index, error]
                        tpl = tuple(closest_index)
                        closest_idx_lbl[tpl] = lbl
                        if not tpl in closest_idx_lbls:
                            closest_idx_lbls[tpl] = [lbl]
                        else:
                            closest_idx_lbls[tpl].append(lbl)

        
        facet_label_index = [
[label_index[0], list(retrieve_original_index(label_index[1], dict_index))]
    for label_index in facet_label_index
]
        
        with open(f'{self.path_f}/facet_label_index.json', 'w') as f:
            json.dump(list(facet_label_index), f, indent=2)
        
        ## Calculate the rotation matrix

        basis_vectors=np.eye(3)

        rotated_basis_vectors=[]
        for vect in basis_vectors:
            rotated_vect=copy.deepcopy(vect)
            rotated_vect = np.dot(rotation_x(alpha1_x), rotated_vect)
            rotated_vect = np.dot(rotation_y(alpha1_y), rotated_vect)
            rotated_vect = np.dot(rotation_z(alpha1_z), rotated_vect)

            rotated_vect = np.dot(rotation(inter_dir_top_facet,
                                           optimal_angle2[0]
                                  ), rotated_vect
            )
            rotated_basis_vectors.append(rotated_vect)

        rotation_matrix = np.array(np.real(rotated_basis_vectors)).T
                                            
        edge_label_index = []
        for i in range(len(facet_label_index)):
            lbl = facet_label_index[i][0]
            if lbl in list(np.unique(edge_label)):
                edge_label_index.append(facet_label_index[i])
                                            
        for lbl in list(np.unique(edge_label)):
            if lbl >= self.nb_facets + 3:
                
                direction = smooth_dir_e[int(lbl)]
                new_direction = list(np.dot(rotation_matrix, direction))
                
                nghbs_label = defaultdict(int)
                for i, j, k in zip(*np.nonzero(edge_label == lbl)):
                    # Define 27 possible offsets
                    offsets = [(dx, dy, dz) 
                               for dx in [-1, 0, 1] 
                               for dy in [-1, 0, 1] 
                               for dz in [-1 , 0, 1]
                    ]
                    for offset in offsets:
                        adj_voxel = tuple(x + y 
                                          for x, y in zip(np.array([i,j,k]), 
                                                          offset
                                                      )
                                    )
                        label_adj_voxel = facet_label[adj_voxel]
                        if label_adj_voxel > 0:
                            nghbs_label[label_adj_voxel] += 1     

                label_1 = max(nghbs_label, key = lambda k: nghbs_label[k])
                nghbs_label[label_1]=0
                label_2 = max(nghbs_label, key = lambda k: nghbs_label[k])
                indice_1 = [facet_label_index[i][1] 
                            for i in range(len(facet_label_index)) 
                            if facet_label_index[i][0] == label_1]
                indice_2 = [facet_label_index[i][1] 
                            for i in range(len(facet_label_index)) 
                            if facet_label_index[i][0] == label_2]
        
                edge_label_index.append([lbl, new_direction, 
                                              [label_1, label_2], 
                                              [list(indice_1), 
                                               list(indice_2)
                                              ]
                                             ]
                )

        corner_label_index = []
        for i in range(len(edge_label_index)):
            lbl = edge_label_index[i][0]
            if lbl in list(np.unique(corner_label)):
                corner_label_index.append(edge_label_index[i])
        
        for lbl in list(np.unique(corner_label)):
            if lbl >= self.nb_facets + 3 + nb_edges + 2:
                direction = smooth_dir_c[int(lbl)]
                new_direction = list(np.dot(rotation_matrix, direction))
                corner_label_index.append([lbl, new_direction])
        
                                            
        with open(f'{self.path_f}/edge_label_index.json', 'w') as f:
            json.dump(list(edge_label_index), f, indent=2) 

        with open(f'{self.path_f}/corner_label_index.json', 'w') as f:
            json.dump(list(corner_label_index), f, indent=2) 

        print('rotation_matrix :', '\n', rotation_matrix, '\n')
        
        nanoscult_matrix = np.copy(rotation_matrix)
        nanoscult_matrix = np.linalg.inv(nanoscult_matrix)
        nanoscult_matrix = np.dot(rotation_y(np.pi/2), nanoscult_matrix)
        print('matrix for nanoscult 90°:')
        print('vector1  ', 
              nanoscult_matrix[0][0], 
              nanoscult_matrix[0][1], 
              nanoscult_matrix[0][2]
        )
        print('vector2  ', 
              nanoscult_matrix[1][0], 
              nanoscult_matrix[1][1], 
              nanoscult_matrix[1][2]
        )
        print('vector3  ', 
              nanoscult_matrix[2][0], 
              nanoscult_matrix[2][1], 
              nanoscult_matrix[2][2]
        )
        print('\n')

        nanoscult_matrix = np.dot(rotation_y(np.pi/2), nanoscult_matrix)
        print('matrix for nanoscult 180°:')
        print('vector1  ', 
              nanoscult_matrix[0][0], 
              nanoscult_matrix[0][1], 
              nanoscult_matrix[0][2]
        )
        print('vector2  ', 
              nanoscult_matrix[1][0], 
              nanoscult_matrix[1][1], 
              nanoscult_matrix[1][2]
        )
        print('vector3  ', 
              nanoscult_matrix[2][0], 
              nanoscult_matrix[2][1], 
              nanoscult_matrix[2][2]
        )
        print('\n')
        
        with open(f'{self.path_f}/facet_label_index.json', 'w') as f:
            json.dump(list(facet_label_index), f, indent=2)
        with open(f'{self.path_f}/edge_label_index.json', 'w') as f:
            json.dump(list(edge_label_index), f, indent=2)
        with open(f'{self.path_f}/corner_label_index.json', 'w') as f:
            json.dump(list(corner_label_index), f, indent=2)
        
        np.save(f'{self.path_f}/rotation_matrix.npy', rotation_matrix)
        np.save(f'{self.path_f}/nanoscult_matrix.npy', nanoscult_matrix)
        
        return facet_label_index, edge_label_index, corner_label_index
      

    def def_mean_strain(self, strain, facet_label, edge_label, corner_label):               
        
        name_list = ['facet', 'edge', 'corner']

        for label_index_name in name_list:
    
            if label_index_name == 'facet':
                fec_label = facet_label
            if label_index_name == 'edge':
                fec_label = edge_label
            if label_index_name == 'corner':
                fec_label = corner_label

            path = (f'{self.path_f}/{label_index_name}'
                        f'_label_index.json'
            )

            with open(path, 'r') as f:
                label_index = json.load(f)
    
            ## Calculate the averaged strain per label 

            for i in range(len(label_index)):
                lbl = label_index[i][0]
                strain_label = 0
                n = 0
                for x,y,z in zip(*np.nonzero(fec_label == lbl)):
                    strain_label += strain[x,y,z]
                    n += 1
                strain_label = strain_label/n
 
                try:
                    if (label_index[i][-1][0] == 'strain'):
                        if not (label_index[i][-1][1] == strain_label):
                            label_index[i][-1][1] = strain_label
                    else: 
                        label_index[i] = (label_index[i] 
                                          + [['strain', strain_label]]
                        )
                except: 
                    label_index[i] = (label_index[i] 
                                      + [['strain', strain_label]]
                    )
    
            for i in range(len(label_index)):
                lbl = label_index[i][0]
                strain_label = label_index[i][-1][1]
                strain_min = 1
                strain_max = 0
                standard_deviation = 0
                n = 0
                for x,y,z in zip(*np.nonzero(fec_label == lbl)):
                    standard_deviation += (strain[x,y,z] - strain_label)**2
                    if strain[x,y,z]<strain_min: 
                        strain_min = strain[x,y,z]
                    if strain[x,y,z]>strain_max: 
                        strain_max = strain[x,y,z]
                    n += 1
                    
                standard_deviation = np.sqrt(standard_deviation/n)
        
                try:
                    if not (label_index[i][-1][2] == standard_deviation):
                        label_index[i][-1][2] = standard_deviation
                except: 
                    label_index[i][-1] = (label_index[i][-1] 
                                          + [standard_deviation]
                    )
                try:
                    if not (label_index[i][-1][3] == strain_max-strain_min):
                        label_index[i][-1][2] = strain_max-strain_min
                except: 
                    label_index[i][-1] = (label_index[i][-1] 
                                          + [strain_max-strain_min]
                    )
    
            if label_index_name == 'facet':
                label_index = sorted(label_index, key= lambda elem : elem[1])
    
            with open(path, 'w') as f:
                json.dump(label_index, f, indent=2) 


    def visualisation(self, facet_label_index, edge_label_index, 
                      corner_label_index, facet_label, edge_label, 
                      corner_label
    ):
                
        def create_3d_view_frames(c_label, 
                                  c_label_index, 
                                  angle, 
                                  c):

            xmin, xmax = np.inf, 0
            ymin, ymax = np.inf, 0
            zmin, zmax = np.inf, 0
            for x, y, z in zip(*np.nonzero(c_label)):
                if x<xmin:
                    xmin = x
                if x>xmax:
                    xmax=x
                if y<ymin:
                    ymin=y
                if y>ymax:
                    ymax=y
                if z<zmin:
                    zmin=z
                if z>zmax:
                    zmax=z
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect([np.ptp(coord) for coord 
                               in np.where(c_label >= 1)])
            ax.set_axis_off()

            def update(frame, angle):
                if angle == 0:
                    ax.view_init(elev=9*frame, azim=0, roll=0)
                if angle == 1:
                    ax.view_init(elev=0, azim=9*frame, roll=0)
                if angle == 2:
                    ax.view_init(elev=0, azim=90, roll=9*frame)

            # Create a list to store frames for saving as PNG
            

            def animate(frame):

                update(frame, angle)
                ax.cla()  # Clear the previous axes

                # Adjust the limits to simulate zooming out
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.set_zlim([zmin, zmax])
                ax.set_axis_off()

                x, y, z = np.where(c_label >= 1)
                values = c_label[x, y, z]

                scatter = ax.scatter(x, y, z, c=values, 
                                     cmap='hsv', s=self.size
                          )

                for i in range(len(c_label_index)):
                    lbl = c_label_index[i][0]
                    if lbl <= self.nb_facets:
                        index = c_label_index[i][1]
                        if index in self.index_to_display:
                            index_x, index_y, index_z = index
                            c_x = np.mean(np.where(c_label == lbl)[0])
                            c_y = np.mean(np.where(c_label == lbl)[1])
                            c_z = np.mean(np.where(c_label == lbl)[2])

                            ax.text(c_x, c_y, c_z, 
                                    f"({index_x}, {index_y}, {index_z})",
                                    color='black', fontsize=8, 
                                    ha='center', zorder=10)

                if angle == 0:
                    if frame in [0, 10, 20, 30]:
                        plt.savefig(f'{self.path_visu}/{c}_{angle}_{frame}.png',
                                    bbox_inches='tight')
                elif angle == 1:
                    if frame in [10, 30]:
                        plt.savefig(f'{self.path_visu}/{c}_{angle}_{frame}.png',
                                    bbox_inches='tight')
                    

            anim = FuncAnimation(fig, animate, frames=40, 
                                 interval=100, blit=False)
            anim.save(f'{self.path_visu}/{c}_gif_{angle}.gif', 
                      writer='Pillow', fps=8)

        for angle in range(3):
            if (self.display_f_e_c == 'facet'
               or self.display_f_e_c == 'all'):
                if not (os.path.exists(f'{self.path_visu}/facet_gif_0.gif') 
                and os.path.exists(f'{self.path_visu}/facet_gif_1.gif') 
                and os.path.exists(f'{self.path_visu}/facet_gif_2.gif') 
                and np.array_equal(self.input_parameters[4], 
                              self.previous_input_parameters[4]
                    ) 
            ):
                    with open(f'{self.path_f}facet_label_index.json', 
                              'r') as f:
                        facet_label_index = json.load(f)
                        
                    create_3d_view_frames(facet_label, 
                                          facet_label_index, 
                                          angle, 
                                         'facet')
                    
            if (self.display_f_e_c == 'edge'
               or self.display_f_e_c == 'all'):
                if not (os.path.exists(f'{self.path_visu}/edge_gif_0.gif') 
                and os.path.exists(f'{self.path_visu}/edge_gif_1.gif') 
                and os.path.exists(f'{self.path_visu}/edge_gif_2.gif') 
                and np.array_equal(self.input_parameters[4], 
                              self.previous_input_parameters[4]
                    ) 
            ):
                    with open(f'{self.path_f}edge_label_index.json', 
                              'r') as f:
                        edge_label_index = json.load(f)
                        
                    create_3d_view_frames(edge_label, 
                                          edge_label_index, 
                                          angle, 
                                         'edge')
            if (self.display_f_e_c == 'corner'
               or self.display_f_e_c == 'all'):
                if not (os.path.exists(f'{self.path_visu}/corner_gif_0.gif') 
                and os.path.exists(f'{self.path_visu}/corner_gif_1.gif') 
                and os.path.exists(f'{self.path_visu}/corner_gif_2.gif') 
                and np.array_equal(self.input_parameters[4], 
                              self.previous_input_parameters[4]
                    ) 
            ):
                    with open(f'{self.path_f}corner_label_index.json', 
                              'r') as f:
                        corner_label_index = json.load(f)
                    create_3d_view_frames(corner_label, 
                                          corner_label_index, 
                                          angle, 
                                         'corner')

### Pipeline 

    def facet_analysis(self) -> np.ndarray :
        
        self.check_previous_data()

        smooth_dir = self.def_smooth_supp_dir()
        edges = None
        if self.remove_edges:
            edges = self.find_edges(smooth_dir)
        cluster_list, cluster_pos = self.def_list_facet_analysis(smooth_dir, 
                                                                 edges
                                    )
        label = self.def_label(cluster_list, cluster_pos, edges)
        facet_label = self.clustering(smooth_dir, label)
        edge_label, corner_label, nb_edges = self.def_edge_corner(facet_label)
        (smooth_dir_f, 
         smooth_dir_e, 
         smooth_dir_c) = self.def_smooth_dir_facet_edge_corner(smooth_dir, 
                                                               facet_label, 
                                                               edge_label, 
                                                               corner_label
                        )
        if (os.path.exists(f'{self.path_f}/facet_label_index.json') 
            and os.path.exists(f'{self.path_f}/edge_label_index.json') 
            and os.path.exists(f'{self.path_f}/corner_label_index.json') 
            and os.path.exists(f'{self.path_f}/rotation_matrix.npy')
        ):

            with open(f'{self.path_f}/facet_label_index.json', 'r') as f:
                facet_label_index = json.load(f)
            with open(f'{self.path_f}/edge_label_index.json', 'r') as f:
                edge_label_index = json.load(f)
            with open(f'{self.path_f}/corner_label_index.json', 'r') as f:
                corner_label_index = json.load(f)

        elif not (os.path.exists(f'{self.path_f}/facet_label_index.json') 
            and os.path.exists(f'{self.path_f}/edge_label_index.json') 
            and os.path.exists(f'{self.path_f}/corner_label_index.json') 
            and os.path.exists(f'{self.path_f}/rotation_matrix.npy')
        ):
            print('No previous facet_label_index, '
                  'edge_label_index or corner_label_index, '
                  'or rotation_matrix found')
            (facet_label_index, 
             edge_label_index, 
             corner_label_index) = self.def_index(smooth_dir_f, 
                                                  smooth_dir_e, 
                                                  smooth_dir_c, 
                                                  facet_label, 
                                                  edge_label, 
                                                  corner_label, 
                                                  nb_edges
                                    )

        elif os.path.exists(f'{self.path_f}/rotation_matrix.npy'):
            rotation_matrix = np.load(f'{self.path_f}/rotation_matrix.npy')
            print('rotation_matrix :', '\n', rotation_matrix, '\n')    
            nanoscult_matrix = np.copy(rotation_matrix)
            nanoscult_matrix = np.linalg.inv(nanoscult_matrix)
            nanoscult_matrix = np.dot(rotation_y(np.pi/2), nanoscult_matrix)
            print('matrix for nanoscult 90°:')
            print('vector1  ', 
                  nanoscult_matrix[0][0], 
                  nanoscult_matrix[0][1], 
                  nanoscult_matrix[0][2]
            )
            print('vector2  ', 
                  nanoscult_matrix[1][0], 
                  nanoscult_matrix[1][1], 
                  nanoscult_matrix[1][2]
            )
            print('vector3  ', 
                  nanoscult_matrix[2][0], 
                  nanoscult_matrix[2][1], 
                  nanoscult_matrix[2][2]
            )
            print('\n')

            nanoscult_matrix = np.dot(rotation_y(np.pi/2), nanoscult_matrix)
            print('matrix for nanoscult 180°:')
            print('vector1  ', 
                  nanoscult_matrix[0][0], 
                  nanoscult_matrix[0][1], 
                  nanoscult_matrix[0][2]
            )
            print('vector2  ', 
                  nanoscult_matrix[1][0], 
                  nanoscult_matrix[1][1], 
                  nanoscult_matrix[1][2]
            )
            print('vector3  ', 
                  nanoscult_matrix[2][0], 
                  nanoscult_matrix[2][1], 
                  nanoscult_matrix[2][2]
            )
            print('\n')
      
        print('number of facet expected = ', self.nb_facets )
        print('number of facet used = ', 
              len(np.unique((facet_label) * self.surface))-1, '\n'
        )
        
        h5_file_path = (f'{self.dump_dir}/cdiutils_S'
                        f'{self.params["metadata"]["scan"]}'
                        f'_structural_properties.h5'
        )
        try:
            with h5py.File(h5_file_path, 'r') as file:
                # Check if 'volume' dataset exists in the file
                if 'volumes' in file:
                    volumes_dataset = file['volumes']
                    if 'het_strain' in volumes_dataset:
                        strain = np.array(volumes_dataset['het_strain'])
                        print('Strain array loaded successfully.', '\n')
                        self.def_mean_strain(strain, 
                                             facet_label, 
                                             edge_label, 
                                             corner_label
                        )
                    else:
                        print('The dataset "het_strain" does not exist '
                              'in the "volume" dataset.', '\n'
                        )
                else:
                    print(f'The dataset "volume" does not exist '
                          f'in {file}.', '\n')     
        except: 
            print(f'The file {h5_file_path} was not found.'
                  f'The mean strain per label was not calculated.'
            )
        self.visualisation(facet_label_index, 
                           edge_label_index, 
                           corner_label_index, 
                           facet_label, 
                           edge_label, 
                           corner_label
        )
  
          



