import numpy as np
import matplotlib.pyplot as plt

def distance(a, b):
    return np.linalg.norm(a - b)

def dist_matrix(a, b):
    c = (a - b).flatten()
    return np.linalg.norm(c)

def get_similarity_matrix(v):
    n = v.shape[0]
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s[i, j] = -distance(v[i], v[j])
    return s

def set_diag(m, v):
    for i in range(m.shape[0]):
        m[i, i] = v[i]
        
def get_coords(dimension, vs):
    
    coordinates = []
    for v in vs:
        coordinates.append(v[dimension])
    return coordinates

def cluster_given_similarity_matrix(s, max_iter, conv_iter, damping, conv_threshold, pref, mode_term, lim_exemp):
    
    n = s.shape[0]

    if pref == 'default':
        set_diag(s, np.median(s) * np.ones(n))
    else:
        set_diag(s, pref)
    
    a = np.zeros((n, n))
    r = np.zeros((n, n))
    indices = np.arange(n)
    
    d = np.zeros((n, n))
    d_old = np.zeros((n, n))
    dist = -np.inf
    dist_old = np.inf
    distances = []
    n_it = 0
    sign_ex = np.zeros(1)
    sign_ex_old = np.zeros(1)
    n_stab_it = 0
    clusters = []
    clusters_old = []
    
    # alternating massage passing
    for it in range(max_iter):
        
        # records the number of iterations
        n_it += 1
        
        # updates responsibility matrix
        aux = np.add(a, s)
        largest_elems = np.max(aux, axis = 0)
        largest_elems_ind = np.argmax(aux, axis = 0)
        aux[indices, largest_elems_ind] = -np.inf
        second_largest_elems = np.max(aux, axis = 0)
        aux = np.tile(largest_elems, (n, 1)).T
        aux[indices, largest_elems_ind] = second_largest_elems
        r = (r + s - aux) * damping

        # updates availability matrix
        aux = np.maximum(r, 0)
        set_diag(aux, np.diag(r))
        aux -= np.sum(aux, axis = 1)
        temp = np.diag(aux).copy()
        aux = np.maximum(aux, 0)
        set_diag(aux, temp)
        a = (a - aux) * damping
        
        # identifies exemplars and clusters
        sign_ex_old = sign_ex
        clusters_old = clusters
        sign_ex = np.sign((np.diag(a) + np.diag(r)) - lim_exemp)
        ind_ex = indices[sign_ex > 0]
        clusters = []
        if len(ind_ex) > 0:
            det = np.add(a, r)
            k = 0
            for i in indices:
                if i not in ind_ex:
                    clusters.append([0, np.argmax(det[i, ind_ex])])
                else:
                    clusters.append([1, k])
                    k += 1
        
        #breaks from iteration if converged
        if mode_term == 'clusters':
            if clusters_old == clusters and len(clusters) > 0:
                n_stab_it += 1
            if n_stab_it == conv_iter:
                break
        elif mode_term == 'matrices':
            d_old = d
            d = np.add(a, r)
            distances.append(dist_matrix(d_old, d))
            if len(distances) >= conv_iter:
                to_break = True
                for i in range(conv_iter):
                    if abs(distances[len(distances) - i - 1]) > conv_threshold:
                        to_break = False
                if to_break:
                    break
            
    return clusters, len(ind_ex), n_it

# the default pref is the median of all similarities
# the default mode of termination is termination after stabilization of clusters
# an alternative mode of termination is termination after stabilization of
# the availability and responsibility matrices
def cluster(data, max_iter = 200, conv_iter = 20, damping = 0.5, conv_threshold = 1e-3, pref = 'default', mode_term = 'clusters', lim_exemp = 1e-3):
    
    clusters = []
    centers = []
    labels = []
    
    s = get_similarity_matrix(data)
    results = cluster_given_similarity_matrix(s, max_iter, conv_iter, damping, conv_threshold, pref, mode_term, lim_exemp)
    n_clusters = results[1]

    for i in range(n_clusters):
        sublist = []
        clusters.append(sublist)

    i = 0
    for item in results[0]:
        labels.append(item[1])
        clusters[item[1]].append(data[i])
        if item[0] == 1:
            centers.append(data[i])
        i += 1
    
    return clusters, centers, results[2], labels