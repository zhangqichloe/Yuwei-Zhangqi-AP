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

def cluster_given_similarity_matrix(s, max_iter, conv_threshold, pref):
    
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
    tol_conv = 0.05
    n_it = 0
    
    # alternating massage passing
    for it in range(max_iter):
        
        # updates responsibility matrix
        aux = np.add(a, s)
        largest_elems = np.max(aux, axis = 0)
        largest_elems_ind = np.argmax(aux, axis = 0)
        aux[indices, largest_elems_ind] = -np.inf
        second_largest_elems = np.max(aux, axis = 0)
        aux = np.tile(largest_elems, (n, 1)).T
        aux[indices, largest_elems_ind] = second_largest_elems
        r = (r + s - aux) * 0.5

        # updates availability matrix
        aux = np.maximum(r, 0)
        set_diag(aux, np.diag(r))
        aux -= np.sum(aux, axis = 1)
        temp = np.diag(aux).copy()
        aux = np.maximum(aux, 0)
        set_diag(aux, temp)
        a = (a - aux) * 0.5
        
        #breaks from iteration if converged
        d_old = d
        d = np.add(a, r)
        dist_old = dist
        dist = dist_matrix(d_old, d)
        if abs(dist_old - dist) < tol_conv:
            break
            
        n_it += 1

    # identifies examplars
    tol = conv_threshold
    sign_ex = np.sign((np.diag(a) + np.diag(r)) - tol)
    ind_ex = indices[sign_ex > 0]
    #print(abs(dist_old - dist))
    
    # identifies clusters
    det = np.add(a, r)
    clusters_view = []
    k = 0
    for i in indices:
        if i not in ind_ex:
            clusters_view.append([0, np.argmax(det[i, ind_ex])])
        else:
            clusters_view.append([1, k])
            k += 1
            
    return clusters_view, len(ind_ex), n_it

def cluster(data, max_iter, conv_threshold, pref):
    
    clusters = []
    centers = []
    labels = []
    
    s = get_similarity_matrix(data)
    results = cluster_given_similarity_matrix(s, max_iter, conv_threshold, pref)
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