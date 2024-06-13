#!/usr/bin/env python3
"""
This file is part of the IIsy project.
This program is a free software tool, which does hybrid in-network machine learning.
licensed under Apache-2.0

Copyright (c) VMware Research(by Broadcom) & Computing Infrastructure Group, Department of Engineering Science, University of Oxford
E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com (no expiration date)

Created on 19.11.2019
@author: Shay Vargaftik, Changgang Zheng
"""

###############################################################################
###############################################################################
import sys
sys.path.append('../multiclass-rade/')

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, make_scorer, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from src.Iris_dataset import load_data
from src.switchML_classifier import switchMLClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
from subprocess import call
import pickle
import os
import logging
import warnings
warnings.filterwarnings("ignore")
from src._forest_selected_features import RandomForestClassifier_Selected_Features as RandomForestClassifier

# Pre-requisites:
# sudo apt-get install graphviz

###############################################################################
###############################################################################
def plotDT(estimator, feature_names, class_names, filename='RF'):
    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot',
                    feature_names = feature_names,
                    class_names = class_names,
                    rounded = True, proportion = False,
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    call(['dot', '-Tpng', 'tree.dot', '-o', filename+'.png', '-Gdpi=600'])

    # Display in jupyter notebook
    Image(filename = filename+'.png')
    return filename+'.png'

def predict(clf, switch_test_confidence):
    clf.set_switch_test_confidence(switch_test_confidence)

    clf_res = clf.predict(X_test, model='hybrid')
    clf_con = clf.predict_proba(X_test, model='hybrid')

    compare_results = clf.compare_switch_server(X_test, y_test)

    auc = roc_auc_score(y_test, clf_con, multi_class = 'ovr')
    mf1 = f1_score(y_test, clf_res, average='macro', zero_division=0)
    accuracy = accuracy_score(y_test, clf_res)

    classification_report_res = classification_report(y_test, clf_res, output_dict=True, zero_division=0)

    precision = precision_score(y_test, clf_res, pos_label=0, zero_division=0, average='macro')
    recall = recall_score(y_test, clf_res, pos_label=0, zero_division=0, average='macro')

    switch_accuracy = compare_results['switch_accuracy']
    server_accuracy = compare_results['server_accuracy']

    switch_test_fraction = clf.get_switch_test_fraction()

    return auc, mf1, accuracy, precision, recall, switch_accuracy, server_accuracy, classification_report_res, switch_test_fraction



if __name__ == "__main__":
    
    folderName = 'IIsy'
    n_jobs = 8
    load_model = False
    plot_metric = 'accuracy' # accuracy or precision
    # feature subset can be set either by features indexes (as printed below), or by feature names.
    features_subset = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    # features_subset = None
    # havedisplay = "DISPLAY" in os.environ
    havedisplay = True
    log_filename = 'log/results.log'
    logging.basicConfig(filename=log_filename, filemode='w', format='%(message)s', level=logging.INFO)

    # set switch_filter to either 'all' or 'only_normal'
    # switch_filter = 'only_normal'
    switch_filter = 'all'

    #check_estimator(switchMLClassifier)
            
    # X, y = UNSW_NB15_binary() #mushroom() #krkopt() #seismic() #car() #cmc() #car() #abalone() #communities() #yeast() #mammography() #letter() #covertype() #UNSW_NB15() #kddcup99() #iris() #wine()

    X_train, y_train, X_test, y_test, used_features = load_data(4, './Data')

    print('features = {}'.format(X_train.columns.tolist()))

    switch_rf_params = {
            
                  'random_state': 42,
                  'n_estimators': 5,
                  'max_depth': 8,
                  'max_samples': None,
                  'class_weight': 'balanced',
                  'max_features': None,
                  # Example: train each decision tree with: 'tree_features': ['dsport', 'proto'], ['proto', 'service'], ['service', 'is_sm_ips_ports'], ['is_sm_ips_ports'], ['proto', 'is_sm_ips_ports'], ...
                  'n_jobs': n_jobs,
            }
    
    server_rf_params = {
            
                  'random_state': 42,
                  'n_estimators': 200,
                  'max_leaf_nodes': 10000,
                  'max_samples': None,
                  'class_weight': None,
                  'n_jobs': n_jobs
                       
            }


    params = {

        'server_params' : server_rf_params,
        'switch_params' : switch_rf_params,

    }

    if features_subset:
        cols = X_train.columns
        if isinstance(features_subset[0], int):
            print('fit switch model with features subset: {}'.format(cols[features_subset].tolist()))
        else:
            print('fit switch model with features subset: {}'.format(features_subset))
        params['switch_train_using_feature_subset'] = features_subset
    else:
        print('fit switch model with all features')

    if switch_filter == 'only_normal':
        params['filter'] = 'only_normal'

    if load_model:
        clf = pickle.load(open('log/{}/clf.pickle'.format(folderName), "rb"))
        clf.set_params(**params)
        clf.fit(X_train, y_train, train_only_switch=True)
    else:
        clf = switchMLClassifier()
        clf.set_params(**params)
        clf.fit(X_train, y_train)

    confidence = [x/100 for x in range(50, 102, 2)] #i/25 for i in range(12, 25, 1)] + [1]

    res_f = []
    res_a = []
    res_acc = []
    res_acc_switch = []
    res_acc_server = []
    res_switch_test_fraction = []
    res_classification_report = []
    for switch_test_confidence in confidence:

        auc, mf1, accuracy, precision, recall, switch_accuracy, server_accuracy, classification_report_res, switch_test_fraction = predict(clf, switch_test_confidence)

        logging.info("Switch confidence th={}: AUC {:5f} , Macro-F1  {:5f} , Accuracy {:5f} , Precision {:5f} , Recall {:5f} , Precision switch {:5f} , Precision server {:5f}  , Switch Fraction {:5f}".format(switch_test_confidence, auc, mf1, accuracy, precision, recall,  switch_accuracy, server_accuracy, switch_test_fraction))
        print("Switch confidence th={}: AUC {:5f} , Macro-F1  {:5f} , Accuracy {:5f} , Precision {:5f} , Recall {:5f} , Precision switch {:5f} , Precision server {:5f}  , Switch Fraction {:5f}".format(switch_test_confidence, auc, mf1, accuracy, precision, recall,  switch_accuracy, server_accuracy, switch_test_fraction))

        res_f.append(mf1)
        res_a.append(auc)

        if plot_metric == 'accuracy':
            res_acc.append(accuracy)
            res_acc_switch.append(switch_accuracy)
            res_acc_server.append(server_accuracy)
        elif plot_metric == 'precision':
            res_acc.append(precision)
            res_acc_switch.append(precision_normal_switch)
            res_acc_server.append(precision_normal_server)
        else:
            raise Exception('Unknown plot_metric {}'.format(plot_metric))

        res_classification_report.append(classification_report_res)
        res_switch_test_fraction.append(switch_test_fraction)

    if havedisplay:
        print('Close fig window to proceed')
        if plot_metric == 'accuracy':
            plt.plot(confidence, res_acc_switch, label='switch Normal accuracy')
            plt.plot(confidence, res_acc_server, label='server Normal accuracy')
            plt.plot(confidence, res_acc, label='total accuracy')
        elif plot_metric == 'precision':
            plt.plot(confidence, res_acc_switch, label='switch Normal precision')
            plt.plot(confidence, res_acc_server, label='server Normal precision')
            plt.plot(confidence, res_acc, label='total precision')


        plt.plot(confidence, res_switch_test_fraction, label='Switch test fraction')
        plt.legend()
        plt.xlabel('switch required test confidence')
        plt.ylabel('Score')
        plt.show()


    pickle.dump([res_acc_switch, res_acc_server, res_acc, res_switch_test_fraction, res_classification_report, confidence], open('log/fig_var.pkl'.format(folderName), "wb"))

    # The chosen working point
    switch_test_confidence = 0.95
    auc, mf1, accuracy, precision, recall, switch_accuracy, server_accuracy, classification_report_res, switch_test_fraction = predict(clf, switch_test_confidence)

    logging.info(
        "Switch confidence th={}: AUC {:5f} , Macro-F1  {:5f} , Accuracy {:5f} , Precision {:5f} , Recall {:5f} , Precision switch {:5f} , Precision server {:5f}   , Switch Fraction {:5f}".format(
            switch_test_confidence, auc, mf1, accuracy, precision, recall,  switch_accuracy, server_accuracy, switch_test_fraction))

    print(
        "Switch confidence th={}: AUC {:5f} , Macro-F1  {:5f} , Accuracy {:5f} , Precision {:5f} , Recall {:5f} , Precision switch {:5f} , Precision server {:5f}  , Switch Fraction {:5f}".format(
            switch_test_confidence, auc, mf1, accuracy, precision, recall, switch_accuracy, server_accuracy, switch_test_fraction))

    switch_model_clf = clf.get_switch_model()

    if not os.path.exists(folderName):
        os.makedirs(folderName)

    filename = 'log/{}-clf-'.format(folderName, 'switch-model')
    pickle.dump(switch_model_clf, open('log/switch_model_clf.pickle'.format(folderName), "wb"))
    pickle.dump(clf, open('log/clf.pickle'.format(folderName), "wb"))


