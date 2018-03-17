# Copyright 2018 Sanket Kamthe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from scipy.linalg import cholesky, solve, solve_triangular
import numpy as np

JIT = 1e-6

def jittered_chol(a, jitter=None, lower=True, overwrite_a=False, check_finite=True):
    if jitter is None:
        jitter = JIT

    a = (a + a.T)/2
    a = a + jitter * np.eye(a.shape[-1])
    chol_a = cholesky(a, lower=lower,
                      overwrite_a=overwrite_a,
                      check_finite=check_finite)

    return chol_a


def jittered_solve(a, b, jitter=None, overwrite_a=False, overwrite_b=False, assume_a='gen', transposed=False):
    if jitter is None:
        jitter = JIT
    a = (a + a.T) / 2
    a = a + jitter * np.eye(a.shape[-1])
    x = solve(a, b, overwrite_a=overwrite_a, overwrite_b=overwrite_b, assume_a=assume_a, transposed=transposed)
    return x


if __name__=="__main__":
    np.random.RandomState(seed=100)
    np.random.seed(seed=100)
    A = np.random.randn(5, 5)
    mat = A @ A.T

    L = jittered_chol(mat, jitter=1e-6)
    np.testing.assert_allclose(mat, L @ L.T, rtol=JIT * 10)