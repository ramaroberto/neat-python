from __future__ import print_function
#import os

import neat
import numpy as np

# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

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
    
def build_tcs(headers, cases):
    class ObjectDict:
        def __init__(self, **entries):
            self.__dict__.update(entries)
        def __repr__(self): 
            return '<%s>' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict.iteritems()))
        
    tcds = []
    for case in cases:
        tcd = {}
        for i, header in enumerate(headers):
            tcd[header] = case[i]
        tcds.append(ObjectDict(**tcd))
    return tcds

def generate_random_sample(quantity, in_dims, func=None, bounds=[-3., 3.]):
    samples = np.random.uniform(bounds[0], bounds[1], (quantity, in_dims))
    if func:
        cols = []
        for i in range(samples.shape[1]):
            cols.append(samples[:, i])
        observations = np.matrix(func(*cols), dtype=np.float64)
        return samples, observations
    return samples

def assertEqualMatrix(tc, m1, m2, precision=1e-20):
    if m1.shape != m2.shape:
        print("Matrices shape do not match:", m1.shape, m2.shape)
    tc.assertTrue(m1.shape == m2.shape)
    result = np.allclose(m1, m2, atol=precision)
    if not result:
        print("Difference found between matrices")
        diff = np.abs(m1 - m2)
        print(diff)
        print("Average diff:", diff.sum()/(diff.shape[0]*diff.shape[1]))
    tc.assertTrue(result)
    
def to_array(m):
    return np.asarray(m).reshape(-1)

if __name__ == '__main__':
    test_softmax()
