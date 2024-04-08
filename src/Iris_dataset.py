"""
This file is part of the IIsy project.
This program is a free software tool, which does hybrid in-network machine learning.
licensed under Apache-2.0
"""

# THIS FILE IS PART OF Planter PROJECT, all Planter's data loader can be easily port to IIsy roject
# Planter.py - The core part of the Planter library
#
# THIS PROGRAM IS FREE SOFTWARE TOOL, WHICH MAPS MACHINE LEARNING ALGORITHMS TO DATA PLANE, IS LICENSED UNDER Apache-2.0
# YOU SHOULD HAVE RECEIVED A COPY OF WTFPL LICENSE, IF NOT, PLEASE CONTACT THE FOLLOWING E-MAIL ADDRESSES
#
# Copyright (c) 2020-2021 Changgang Zheng
# Copyright (c) Computing Infrastructure Lab, Departement of Engineering Science, University of Oxford
# E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(num_features, data_dir):
    iris = pd.read_csv(data_dir+'/Iris/Iris.csv', usecols=[1, 2, 3, 4, 5])

    for label in iris:
        if label =='Species':
            break
        iris[label] = (iris[label]*10).astype("int")
    used_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'][:num_features]
    X = iris[used_features]
    y = iris['Species']


    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    # print(y)


    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 101, shuffle=True)
    return train_X, train_y, test_X, test_y, used_features