#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

import pytest
from pyomo.common import unittest as pyo_unittest
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.Time_Series_Clustering import TimeSeriesClustering
import idaes.logger as idaeslog
import os
import numpy as np

current_path = os.getcwd()

@pytest.fixture
def test_simultaion_data():
    return os.path.join(current_path, 'tests','test_data', 'simdatatest.csv')


@pytest.fixture
def test_input_data():
    return os.path.join(current_path, 'tests','test_data', 'inputdatatest.h5')


@pytest.fixture
def num_sims():
    return 3


@pytest.fixture
def case_type():
    return 'NE'


@pytest.fixture
def fixed_pmax():
    return True


@pytest.fixture
def num_clusters():
    return 1

@pytest.fixture
def filter_opt():
    return True

@pytest.fixture
def metric():
    return 'euclidean'


@pytest.fixture
def test_clustering_results():
    return os.path.join(current_path, 'tests','test_data', 'test_clustering_model.json')


@pytest.fixture
def base_simulationdata(test_simultaion_data, test_input_data, num_sims, case_type, fixed_pmax):
    return SimulationData(test_simultaion_data, test_input_data, num_sims, case_type, fixed_pmax)


@pytest.fixture
def base_timeseriesclustering(num_clusters, base_simulationdata, filter_opt, metric):
    return TimeSeriesClustering(num_clusters, base_simulationdata, filter_opt, metric)


@pytest.mark.unit
def test_create_TimeSeriesClustering(num_clusters, base_simulationdata, filter_opt, metric):
    tsc = TimeSeriesClustering(num_clusters, base_simulationdata, filter_opt, metric)
    assert tsc.num_clusters is num_clusters
    assert tsc.simulation_data is base_simulationdata
    assert tsc.filter_opt is filter_opt
    assert tsc.metric is metric


@pytest.mark.unit
def test_transform_data(base_timeseriesclustering):
    train_data = base_timeseriesclustering._transform_data()
    # test on the shape of the data to see if the filter is working. 
    data_shape = np.shape(train_data)
    expect_data_shape = (366,24,1)

    pyo_unittest.assertStructuredAlmostEqual(
        first=data_shape, second=expect_data_shape
    )


@pytest.mark.unit
def test_get_cluster_centers(base_timeseriesclustering, test_clustering_results):
    centers_dict = base_timeseriesclustering.get_cluster_centers(test_clustering_results)
    a = []
    for i in range(24):
        a.append([0.25])
    
    expected_centers_dict = {0:np.array(a)}

    for key_a, key_b in zip(centers_dict,expected_centers_dict):

        pyo_unittest.assertStructuredAlmostEqual(
            first=key_a, second=key_b
        )
        np.testing.assert_array_equal(centers_dict[key_a], expected_centers_dict[key_b])