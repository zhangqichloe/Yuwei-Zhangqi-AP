# Yuwei-Zhangqi-AP

Documentation for APClustering

This clustering method implements "affinity propagation". The main function is cluster(data, max_iter, conv_iter, damping, conv_threshold, pref, mode_term, lim_exemp). While other functions are callable, in practice they do not have much of a use except in the execution of the main function.

The parameters of cluster():
    
    data: this is the data to be classfied.

    max_iter: the maximum number of iterations; default is 200
    
    conv_iter: ideally in all cases of mode_term, this is the minimum number of iterations a condition has to stay stable for the iterations to stop; default: 15
        
    damping: the damping ratio, representing the proportion of information kept when updating; default: 0.5
        
    conv_threshold: stability criteria for the 'matrices' mode of termination, the more it is increased, the less the matrices have to be similar to stop the iterations; default: 1e-3
    
    pref: the preference vector, the diagonal of the similarity matrix; default: the median of all entries in the similarity matrix, except the ones on the diagonal

    mode_term: the criteria for stopping the loop, currently only containing 'clusters' and 'matrices'; default: 'clusters'
        
    lim_exemp: the lower limit of an entry in the diagonal of the determination matrix (responsibility + availability) for the corresponding data point to qualify as exemplar; default: 1e-3
      
cluster() will return four values:

    clusters: a collection of collections of data points (in vector form), with each of the sub-collections representing a cluster.
    
    centers: a collection of data points (in vector form), with each point being an exemplar
    
    n_it: number of iterations
    
    labels: a collection of integers, each representing which cluster the corresponding data point (corresponding with respect to the order of points in the first returned value) belongs to
        
        

References

Brendan J. Frey and Delbert Dueck, “Clustering by Passing Messages Between Data Points”, Science Feb. 2007
