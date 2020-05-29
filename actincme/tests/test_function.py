#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example of a test file using a function.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""

import pytest
from actincme.bin.filament import Filament
from actincme.bin.symmetricize import Symmetricize
from actincme.bin.rotate import Rotate, AverageRotate
import matplotlib.pyplot as plt
import os
import numpy as np

os.chdir(r'./actincme/tests/data')

# manually determined slices
start_list = [1, 4, 4, 4, 5, 6, 5, 6, 5, 6, 2, 4, 5, 3, 6, 7, 1, 4, 1, 2, 1, 5, 1, 1, 1, 0, 1]
end_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -3, -1, -1, -1, -1, -1, -1, -2, -1, -2, -1]


# If you only have a single condition you need to test, a single test is _okay_ but parametrized tests are encouraged
def test_no_NaNs():
    start_val = 5
    new_val = 20

    for i in range(27):
        if i not in [13, 17, 18, 20, 22, 24, 26]:
            shape = Symmetricize('./', i+1, start_list[i], end_list[i])
            this_x, this_y, this_z = shape.do_everything_2d("fit", plot=False) #handles all the logic in symmetricizing fit curves
            mean_x, mean_y = shape.get_mean_coords()
            this_rotate = Rotate(this_x, this_y, this_z, mean_x, mean_y)    
            x, y, zs = this_rotate.rotate_steps()
            assert ~np.isnan(x).any()
            assert ~np.isnan(y).any()
            assert ~np.isnan(zs).any()

def test_filament_directions():
    """
    Should be between -90 and 90
    """
    i =4
    shape = Symmetricize('./', i+1, start_list[i], end_list[i])
    this_x, this_y, this_z = shape.do_everything_2d("fit", plot=False) #handles all the logic in symmetricizing fit curves
    mean_x, mean_y = shape.get_mean_coords()
    this_rotate = Rotate(this_x, this_y, this_z, mean_x, mean_y)
    test_x, test_y, test_z = this_rotate.rotate_steps()
    filaments = Filament('./', 'BranchedActinCoordinates_Integers')
    filaments.calculate_directionality(rotated_surface=this_rotate) #handles all the logic 
    assert np.array([-91 < e < 91 for e in filaments._filament_orientation_dataframe['zdir_rel'].values]).all()
    assert np.array([-91 < e < 91 for e in filaments._filament_orientation_dataframe['ydir_rel'].values]).all()


# # Generally, you should parametrize your tests, but you should include exception tests like below!
# @pytest.mark.parametrize("start_val, next_val, expected_values", [
#     # (start_val, next_val)
#     (5, 20, (20, 5)),
#     (10, 40, (40, 10)),
#     (1, 2, (2, 1))
# ])
# def test_parameterized_value_change(start_val, next_val, expected_values):
#     example = Example(start_val)
#     example.update_value(next_val)
#     assert expected_values == example.values


# # The best practice would be to parametrize your tests, and include tests for any exceptions that would occur
# @pytest.mark.parametrize("start_val, next_val, expected_values", [
#     # (start_val, next_val)
#     (5, 20, (20, 5)),
#     (10, 40, (40, 10)),
#     (1, 2, (2, 1)),
#     pytest.param("hello", None, None, marks=pytest.mark.raises(exception=ValueError)),  # Init value isn't an integer
#     pytest.param(1, "hello", None, marks=pytest.mark.raises(exception=ValueError))  # Update value isn't an integer
# ])
# def test_parameterized_value_change_with_exceptions(start_val, next_val, expected_values):
#     example = Example(start_val)
#     example.update_value(next_val)
#     assert expected_values == example.values
