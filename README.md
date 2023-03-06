# Random-forests
Ensemble methods are ideal for regression and classification as they reduce bias and variance and increase model accuracy. Ensemble has two techniques sag and boost. This project focuses on bagging techniques used in random forest models.  Splitting of the tensile test is performed. A random forest model is used for modeling. Feature importance is to create a function, find the important features and plot the features. 
Bootstrap:
Whether to use bootstrap sampling when building the tree. If False, we use the entire dataset to build each tree.

Max Depth :
Maximum depth of the tree. If None, expand nodes until all leaves are pure or all leaves contain less than min_samples_split samples.

Min_Samples_Leaf :
Minimum number of samples required in leaf nodes. Arbitrary depth split points are only considered if there are at least min_samples_leaf training samples left in each of the left and right branches. This can lead to smoothing of the model, especially for regression.

Min_Samples_Split :
Minimum number of samples required to share an internal node.

N_estimator :
number of trees in the forest.

Random_State:
Controls both the bootstrap randomness of the samples used to build the tree (if bootstrap=True) and the sampling of features considered when searching for the best split at each node (if max_features < n_features) To do. 
