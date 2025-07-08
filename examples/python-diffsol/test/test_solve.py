from python_diffsol import PyDiffsol
import numpy as np
import unittest


class TestStringMethods(unittest.TestCase):
    def test_solve(self):
        model = PyDiffsol(
            """
            in = [r, k]
            r { 1 } k { 1 }
            u_i { y = 0.1 }
            F_i { (r * y) * (1 - (y / k)) }
            """
        )
        times = np.linspace(0.0, 1.0, 100)
        k = 1.0
        r = 1.0
        y0 = 0.1
        y = model.solve(np.array([r, k]), times)
        soln = k / (1.0 + (k - y0) * np.exp(-r * times) / y0)
        np.testing.assert_allclose(y[0], soln, rtol=1e-5)
        
if __name__ == '__main__':
    unittest.main()