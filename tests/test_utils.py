from __future__ import print_function
#import os
import sys
import warnings

import neat

warnings.simplefilter('default')

# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality,
# particularly for test_random_proportional_selection.

class NotAlmostEqualException(Exception):
    pass


def assert_almost_equal(a, b):
    if abs(a - b) > 1e-6:
        max_abs = max(abs(a), abs(b))
        abs_rel_err = abs(a - b) / max_abs
        if abs_rel_err > 1e-6:
            raise NotAlmostEqualException("{0:.6f} !~= {1:.6f}".format(a, b))


def test_softmax():
    """Test the neat.math_utils.softmax function."""
    # Test data - below is from Wikipedia Softmax_function page.
    test_data = [([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [0.02364054302159139, 0.06426165851049616,
                                                        0.17468129859572226, 0.47483299974438037,
                                                        0.02364054302159139, 0.06426165851049616,
                                                        0.17468129859572226])]

    for test in test_data:
        results_list = list(neat.math_util.softmax(test[0]))
        for a, b in zip(test[1], results_list):
            assert_almost_equal(a, b)

    #softmax_result = list(neat.math_util.softmax([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]))
    #print("Softmax for [1, 2, 3, 4, 1, 2, 3] is {!r}".format(softmax_result))

def test_random_proportional_selection():
    """Test the neat.math_utils.random_proportional_selection function."""

    assert neat.math_util.random_proportional_selection([1.0]) == 0
    assert neat.math_util.random_proportional_selection([0.0,1.0]) == 1
    assert neat.math_util.random_proportional_selection([0.0,0.0,2.0]) == 2
    assert neat.math_util.random_proportional_selection([0.0,0.0001,0.0]) == 1
    assert 0 < neat.math_util.random_proportional_selection([0.0,1.0,1.0]) <= 2

    saw_01 = False
    saw_2 = False
    likelihood = 1.0
    times_tried = 0
    while not (saw_01 and saw_2):
        times_tried += 1
        result = neat.math_util.random_proportional_selection([0.5,0.5,1.0])
        if result == 2:
            saw_2 = True
        elif 0 <= result <= 1:
            saw_01 = True
        else:
            raise AssertionError("Saw bad result {0!r}".format(result))
        likelihood *= 0.5
        if likelihood >= sys.float_info.epsilon:
            raise AssertionError(
                "Tried for both results {0:n} times (saw_01 {1} saw_2 {2}".format(
                    times_tried, saw_01, saw_2))

if __name__ == '__main__':
    test_softmax()
    test_random_proportional_selection()
