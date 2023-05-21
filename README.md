# Gesture Recognition Using Machine Learning: SVM and Random Forest
This code was made as part of the bachelor thesis Gesture Recognition Using Machine Learning in 2023 at Comenius University in Bratislava, Slovakia.
It is a modification of the code received form my supervisor Ing. Viktor Kocur, PhD.

The main files written as part of this thesis are: 
1. experiment_evaluation.py - code executing all the experiments using models SVM and RF in our thesis
2. training_and_testing.py - code performing all training and testing in our thesis on top of loading data and creating sequences and preprocessing
3. results_extraction.py - code for extraction of results from .json files
4. files_organization.py - code that stores and loads from the .json/.joblib files
5. results_visualization.py - code for results visualization
6. frames_visualization.py - code for visualization of frames of gestures

The following files are runnable and can be run in order to conduct all the experiments required for the thesis and generate the results visualization or frames visualization:
1. experiment_evaluation.py
2. results_visualization.py
3. frames_visualization.py

In order for the above runnable files to work, it is necessary to download the following dataset and set and use its location path where needed (experiment_evaluation.py and frames_visualization.py):
```
http://cogsci.dai.fmph.uniba.sk/~kocur/gestures/
```

Results:
1. test_pca - folders containing the results of experiments on combinations of preprocessing for models SVM and Random Forest for different sequence length

2. test_svm_kernels - folder containing the result of kernel test for SVM

3. test_hyper_parameters - folder containing results of hyperparameter tests on the best combinations of preprocessing for both SVM and RF