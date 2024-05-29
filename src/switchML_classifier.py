#!/usr/bin/env python3

# Copyright 2019-2022 VMware, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
This file is part of the IIsy project.

Based on: 
https://github.com/vmware-samples/efficient-multiclass-classification 

Under the BSD-3-Clause license:

Copyright 2022 VMware, Inc.				

The BSD 3-Clause "New" or "Revised" license (the "License") set forth below applies to all parts of the Efficient Multi-class Classification project.  You may not use this file except in compliance with the License.

BSD-3 License 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import time

import numpy as np
import xgboost as xgb

import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.ensemble import RandomForestClassifier
from src._forest_selected_features import RandomForestClassifier_Selected_Features as RandomForestClassifier

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.metrics import precision_score, accuracy_score, classification_report

###############################################################################
###############################################################################
class switchMLClassifier(BaseEstimator, ClassifierMixin):
    """
    A switchML classifier.

    The two main building blocks of switchML are two classifiers.

    The first classifier is a small Coarse-Grained (switch) model - Random Forest
    (RF) that is trained using the entire training dataset and which we use 
    to compute the labeled data predictability*. 

    The second classifier is a Fine-Grained (fg) model - XGBoost that is
    trained using only a predictability-driven fraction of the training 
    dataset. During classification, all data instances are classified by 
    the RF, and only the hard data instances (i.e., cases for which the RF 
    is not sufficiently confident) are forwarded to the XGBoost for
    reclassification.

    *Predictability of an instance is defined to be the distance (l2 norm) 
    among the distribution vector (i.e., predict_proba) by the RF and the 
    'perfect' one (i.e., 1 in the correct class and 0 at all others).
    
    For more information read: Efficient_Multi-Class_Classification_with_Duet.pdf
    
    Parameters
    ----------

    switch_train_using_feature_subset : list or None, optional (default=None)
        List of columns to use (when None, all columns are used).

    duet_fg_train_dataset_fraction : float, optional (default=0.25)
        A value in (0,1]. Indicated the data fraction that is used for the
        training of the fg (XGBoost) model. If duet_subsample_only is set to True,
        indicates the data fraction for dataset sub-sampling.
        
    duet_fg_train_per_class_data_fraction : float, optional (default=0)
        A value in [0,1]. Indicated the per-class data fraction that is used for the
        training of the fg (XGBoost) model. If positive, each class gets 
        duet_fg_train_dataset_fraction*duet_fg_train_per_class_data_fraction/#classes.

    switch_test_confidence : float, optional (default=0.95)
        A value in [0,1]. Indicated the data confidence (i.e., 
        top-1 probability in the distribution vector) above which the instance
        is not passed to the fg (XGBoost) classifier for classification (i.e., classified
        only by the switch (RF) classifier). Used only with duet_fg_test=False.
        
    verbose : boolean, optional (default=False)
        Verbose printing for debug. Print the fraction of the data that is 
        used for the training and classification by the fg (XGBoost) classifier.

    seed : int, optional (default=42)
        Random seed for the numpy package used in switchML.

    switch_params : dict or None, optional (default={'max_leaf_nodes': 1000})
        Parameters for the switch (RF) classifier.
        The default max_leaf_nodes parameter is used to avoid any over-fitting
        by the switch (RF) model.

    server_params : dict or None, optional (default=None)
        Parameters for the fg XGBoost classifier.
                       
    duet_subsample_only : boolean, optional (default=False)
        When true, use duet for sub-sampling of the dataset. Fit returns the sub-sampled dataset
        (X',y'), according to the duet_fg_train_dataset_fraction value.
               
    duet_fg_test : boolean, optional (default=False)
        If true, all data is classified only by the fg (XGBoost) classifier.
    
    Attributes
    ----------

    classes_ : array of shape (n_classes,) classes labels. 

    switch_clf_ : RandomForestClassifier
        The switch (Random Forest) classifier.
        
    fg_clf_ : xgboost
        The fg (XGBoost) classifier.
    
    fg_clf_fitted_ : Boolean
        Remembers if fit was called for the fg model within switchML.
    
    fit_time_, predict_time_ : float
        Temporary. Used for debug measurements

    fg_clf_predict_data_fraction_ : float
        Test dataset fraction which is predicted by the fg_clf


    Example program
    ---------------

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)   
    
    switch_params = {
            
                  'n_estimators': 20,
                  'max_leaf_nodes': 100,
                    
            }
    
    server_params = {
            
                  'n_estimators': 1000,
                  'max_depth': 8,
                  'learning_rate': 0.01,

            }
                    
    parameters = {
                  
                  'duet_fg_train_dataset_fraction': 0.1,
                  'switch_test_confidence': 0.99,
                                   
                  'switch_params': switch_params,
                  'server_params': server_params          

                 }
    
    duet = switchMLClassifier()
    duet.set_params(**parameters)       
    duet.fit(X_train, y_train)   

    y_predicted = duet.predict(X_test)
    print(classification_report(y_test, y_predicted, digits=5))


    Notes
    -----
    
    The parameters controlling the size of the dataset for the fg training
    (duet_fg_train_dataset_fraction) and the switch confidence level
    (switch_test_confidence) are advised to be specifically tuned for each dataset
    (e.g., by grid-search).
    
    The parameters for the RF and XGBoost classifiers should also be tuned.
    Using the parameters that work well for the monolithic models is a good
    start.         
    """

    ###########################################################################
    ###########################################################################

    def __init__(self,

                 ### parameters                 
                 server_model_type='RF',
                 server_params=None,

                 switch_model_type='RF',
                 switch_params=None,
                 switch_train_using_feature_subset=None,
                 switch_test_confidence=0.95,
                 filter='all',
                 seed=42,
                 verbose=False,
                                    
                 ):

        self.server_model_type = server_model_type
        self.server_params = server_params
        
        self.switch_model_type = switch_model_type
        self.switch_params = switch_params
        self.switch_train_using_feature_subset = switch_train_using_feature_subset
        self.switch_test_confidence = switch_test_confidence
        self.filter = filter

        self.seed = seed
        self.verbose = verbose
        
    ###########################################################################
    ###########################################################################

    def verify_parameters(self, X, y):

        if self.switch_test_confidence < 0 or self.switch_test_confidence > 1:
            raise Exception("Illegal switch_test_confidence value. Should be in [0, 1]")

        if self.verbose not in [True, False]:
            raise Exception("Illegal verbose value. Should be in [True, False]")
        
        if self.switch_train_using_feature_subset is not None:

            ### empty is not allowed
            if not len(self.switch_train_using_feature_subset):
                raise Exception("Illegal switch_train_using_feature_subset (err1): {}\nShould be None or specify unique columns".format(self.switch_train_using_feature_subset))
                
            ### duplicates are not allowed
            if len(self.switch_train_using_feature_subset) != len(set(self.switch_train_using_feature_subset)):
                raise Exception("Illegal switch_train_using_feature_subset (err2): {}\nShould be None or specify unique columns".format(self.switch_train_using_feature_subset))
                
            ### translate column names (if X is a dataframe) to indices
            if isinstance(X, pd.DataFrame): 
                if all(elem in X.columns for elem in self.switch_train_using_feature_subset):
                    self.switch_train_using_feature_subset = [X.columns.get_loc(i) for i in self.switch_train_using_feature_subset]
            
            ### verify legal column values                
            if not set(self.switch_train_using_feature_subset).issubset(set(range(X.shape[1]))):
                raise Exception("Illegal switch_train_using_feature_subset (err3): {}\nShould be None or specify unique columns".format(self.switch_train_using_feature_subset))

        if self.switch_model_type != self.server_model_type:
            raise Exception("Illegal model type configuration {} != {}".format(self.server_model_type, self.switch_model_type))

        if self.switch_model_type not in ['RF', 'XGB']:
            raise Exception("Unknown switch model type ({})".format(self.switch_model_type))

    ###########################################################################
    ###########################################################################

    def fit(self, X, y, train_only_switch=False):
        """
        Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples.

        y : array-like, shape=(n_samples,)
            The input sample labels.
            
        Returns
        -------
        self : object
        """
        
        ### set numpy seed
        np.random.seed(self.seed)

        ### performance measurments
        self.fit_time_ = {}
        self.predict_time_ = {}
        
        ### switch predict fraction
        self.switch_predict_fraction_ = 0 
        self.mask_ = None

        ### parameters input checks
        self.verify_parameters(X, y)
            
        ### input verification - required by scikit         
        X, y = check_X_y(X, y)

        ### store the classes seen during fit - required by scikit
        self.classes_ = unique_labels(y)

        ### init switch
        if self.switch_model_type == 'RF':
            self.switch_clf_ = RandomForestClassifier()
        elif self.switch_model_type == 'XGB':
            self.switch_clf_ = xgb.XGBClassifier()
        
        ### set params
        if self.switch_params is None: 
            print("\nWarning: no kwards for the switch model.\n")
        else:
            self.switch_clf_.set_params(**self.switch_params)

        start = time.time()
        if self.switch_train_using_feature_subset == None:
            ### train switch model using all features
            self.switch_clf_.fit(X, y)
        else:
            ### train switch model using features subset specified by self.switch_train_using_feature_subset
            self.switch_clf_.fit(X[:, self.switch_train_using_feature_subset], y)
        end = time.time()
        self.fit_time_['switch'] = end - start
        
        if not train_only_switch:  
            
            ### init server
            if self.server_model_type == 'RF':
                self.server_clf_ = RandomForestClassifier()
            elif self.server_model_type == 'XGB':
                self.server_clf_ = xgb.XGBClassifier()
        
            ### set params
            if self.server_params is None:
                print("\nWarning: no kwards for the server model.\n")
            else:
                self.server_clf_.set_params(**self.server_params)

            start = time.time()
            ### train server model using all features
            self.server_clf_.fit(X, y)
            end = time.time()
            self.fit_time_['server'] = end-start
            

        ### a call to fit should return the classifier - required by scikit
        return self

    ###########################################################################
    ###########################################################################
    
    def predict(self, X, model='server'):
        """
        Predict labels for X rows.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples.
           
        Returns
        -------
        y : nparray of class labels or class distributions 
            for X, shape=(n_samples,) or shape=(n_samples, n_classes).
        """
        
        ### set numpy seed
        np.random.seed(self.seed)

        ### check is that fit had been called - required by scikit
        check_is_fitted(self)

        ### input verification - required by scikit
        X_test = check_array(X)
        
        if model == 'server':
            
            ### predict using the server model and return
            return self.server_clf_.predict(X_test)
        
        elif  model == 'switch':
            
            ### predict using the switch model and return
            if self.switch_train_using_feature_subset == None:
                return self.switch_clf_.predict(X_test)
            else:
                return self.switch_clf_.predict(X_test[:, self.switch_train_using_feature_subset])
        
        elif model == 'hybrid':

            ### predict using the switch model
            if self.switch_train_using_feature_subset == None:
                switch_predic_proba = self.switch_clf_.predict_proba(X_test)
            else:
                switch_predic_proba = self.switch_clf_.predict_proba(X_test[:, self.switch_train_using_feature_subset])

            switch_predictions = self.classes_.take(np.argmax(switch_predic_proba, axis=1), axis=0)

            if self.filter == 'all':
                # original
                switch_classification_confidence = np.max(switch_predic_proba, axis=1)
                mask = switch_classification_confidence <= self.switch_test_confidence
            elif self.filter == 'only_normal':
                # use confidence threshold just for the normal labeles
                switch_classification_normal_confidence = switch_predic_proba[:, 0]
                mask = ~(switch_classification_normal_confidence > max(self.switch_test_confidence, 0.5))
            elif isinstance(self.filter, list):
                masks = []
                for c in self.filter:
                    switch_classification_c_confidence = switch_predic_proba[:, c]
                    masks.append(switch_classification_c_confidence <= self.switch_test_confidence)
                mask = [True] * len(switch_predic_proba)
                for m in masks:
                    mask = [mask_i & m_i for mask_i, m_i in zip(mask, m.tolist())]
                mask = np.asarray(mask)

            self.mask_ = mask

            self.switch_predict_fraction_ = sum(~mask)/len(mask)
            if self.switch_predict_fraction_ < 1:
                switch_predictions[mask] = self.server_clf_.predict(X_test[mask])
                
            return switch_predictions
        
        else:
            raise Exception("unknown model: {}. should be in ['switch','server','hybrid'].".foramt(model))
        

    ###########################################################################
    ###########################################################################

    def predict_proba(self, X, model='server'):
        """
        Predict labels for X rows.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples.
           
        Returns
        -------
        y : nparray of class labels or class distributions 
            for X, shape=(n_samples,) or shape=(n_samples, n_classes).
        """
        
        ### set numpy seed
        np.random.seed(self.seed)

        ### check is that fit had been called - required by scikit
        check_is_fitted(self)

        ### input verification - required by scikit
        X_test = check_array(X)
        
        if model == 'server':
            
            ### predict using the server model and return
            return self.server_clf_.predict_proba(X_test)
        
        elif  model == 'switch':
            
            ### predict using the switch model and return
            if self.switch_train_using_feature_subset == None:
                return self.switch_clf_.predict_proba(X_test)
            else:
                return self.switch_clf_.predict_proba(X_test[:, self.switch_train_using_feature_subset])

        elif model == 'hybrid':

            ### predict using the switch model
            if self.switch_train_using_feature_subset == None:
                switch_predic_proba = self.switch_clf_.predict_proba(X_test)
            else:
                switch_predic_proba = self.switch_clf_.predict_proba(X_test[:, self.switch_train_using_feature_subset])

            if self.filter == 'all':
                # original
                switch_classification_confidence = np.max(switch_predic_proba, axis=1)
                mask = switch_classification_confidence <= self.switch_test_confidence
            elif self.filter == 'only_normal':
                # use confidence threshold just for the normal labeles
                switch_classification_normal_confidence = switch_predic_proba[:, 0]
                mask = ~(switch_classification_normal_confidence > max(self.switch_test_confidence, 0.5))
            elif isinstance(self.filter, list):
                masks = []
                for c in self.filter:
                    switch_classification_c_confidence = switch_predic_proba[:, c]
                    masks.append(switch_classification_c_confidence <= self.switch_test_confidence)
                mask = [True] * len(switch_predic_proba)
                for m in masks:
                    mask = [mask_i & m_i for mask_i, m_i in zip(mask, m.tolist())]
                mask = np.asarray(mask)

            self.mask_ = mask

            self.switch_predict_fraction_ = sum(~mask)/len(mask)
            if self.switch_predict_fraction_ < 1:
                switch_predic_proba[mask] = self.server_clf_.predict_proba(X_test[mask])
                
            return switch_predic_proba
        
        else:
            raise Exception("unknown model: {}. should be in ['switch','server','hybrid'].".foramt(model))
            
    ###########################################################################
    ###########################################################################
    
    def set_switch_test_confidence(self, confidence):
        if confidence < 0 or confidence > 1:
            raise Exception("Illegal confidence value. Should be in [0, 1]")
        self.switch_test_confidence = confidence  
 
    ###########################################################################
    ###########################################################################
    
    def get_switch_test_fraction(self):
        return self.switch_predict_fraction_
    
    def get_switch_model(self):
        return self.switch_clf_

    def get_server_model(self):
        return self.server_clf_
    
    ###########################################################################
    ###########################################################################
    # must be called after predict / predict_proba, such that self.mask is updated.
    def compare_switch_server(self, X, y):
        if self.mask_ is None:
            raise Exception('Must call predict or predict proba before')

        ### input verification - required by scikit
        X_test = check_array(X)

        if any(~self.mask_):
            if self.switch_train_using_feature_subset == None:
                y_pred_switch = self.switch_clf_.predict(X_test[~self.mask_])
            else:
                y_pred_switch = self.switch_clf_.predict(X_test[~self.mask_][:, self.switch_train_using_feature_subset])

            y_pred_server = self.server_clf_.predict(X_test[~self.mask_])

            if len(np.unique(y)) == 2: # binary case
                results = {
                    'y_pred_switch': y_pred_switch,
                    'y_pred_server': y_pred_server,
                    'mask': ~self.mask_,
                    'switch_classification_report': classification_report(y[~self.mask_], y_pred_switch, output_dict=True, zero_division=0),
                    'server_classification_report': classification_report(y[~self.mask_], y_pred_server, output_dict=True, zero_division=0),
                    'switch_classification_report_text': classification_report(y[~self.mask_], y_pred_switch, zero_division=0),
                    'server_classification_report_text': classification_report(y[~self.mask_], y_pred_server, zero_division=0),
                    'switch_normal_precision': precision_score(y[~self.mask_], y_pred_switch, pos_label=0, zero_division=0),
                    'server_normal_precision': precision_score(y[~self.mask_], y_pred_server, pos_label=0, zero_division=0),
                    'switch_anomaly_precision': precision_score(y[~self.mask_], y_pred_switch, pos_label=1, zero_division=0),
                    'server_anomaly_precision': precision_score(y[~self.mask_], y_pred_server, pos_label=1, zero_division=0),
                    'switch_accuracy': accuracy_score(y[~self.mask_], y_pred_switch),
                    'server_accuracy': accuracy_score(y[~self.mask_], y_pred_server),
                }

            else: # multi-class case
                results = {
                    'y_pred_switch': y_pred_switch,
                    'y_pred_server': y_pred_server,
                    'mask': ~self.mask_,
                    'switch_classification_report': classification_report(y[~self.mask_], y_pred_switch, output_dict=True, zero_division=0),
                    'server_classification_report': classification_report(y[~self.mask_], y_pred_server, output_dict=True, zero_division=0),
                    'switch_classification_report_text': classification_report(y[~self.mask_], y_pred_switch, zero_division=0),
                    'server_classification_report_text': classification_report(y[~self.mask_], y_pred_server, zero_division=0),
                    'switch_accuracy': accuracy_score(y[~self.mask_], y_pred_switch),
                    'server_accuracy': accuracy_score(y[~self.mask_], y_pred_server),
                }

        else:
            if len(np.unique(y)) == 2: # binary case
                results = {
                    'y_pred_switch': None,
                    'y_pred_server': None,
                    'mask': ~self.mask_,
                    'switch_classification_report': None,
                    'server_classification_report': None,
                    'switch_classification_report_text': None,
                    'server_classification_report_text': None,
                    'switch_normal_precision': 0,
                    'server_normal_precision': 0,
                    'switch_anomaly_precision': 0,
                    'server_anomaly_precision': 0,
                    'switch_accuracy': 0,
                    'server_accuracy': 0,
                }
            else: # multi-class case
                results = {
                    'y_pred_switch': None,
                    'y_pred_server': None,
                    'mask': ~self.mask_,
                    'switch_classification_report': None,
                    'server_classification_report': None,
                    'switch_classification_report_text': None,
                    'server_classification_report_text': None,
                    'switch_accuracy': 0,
                    'server_accuracy': 0,
                }

        return results
