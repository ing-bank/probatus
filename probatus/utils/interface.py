# Copyright (c) 2020 ING Bank N.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from abc import ABC, abstractmethod
from probatus.utils import NotFittedError


class BaseFitComputeClass(ABC):
    fitted = False

    def _check_if_fitted(self):
        """
        Checks if object has been fitted. If not, NotFittedError is raised.
        """
        if self.fitted is False:
            raise(NotFittedError('The object has not been fitted. Please run fit() method first'))

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit_compute(self, *args, **kwargs):
        pass


class BaseFitComputePlotClass(BaseFitComputeClass):

    @abstractmethod
    def plot(self, *args, **kwargs):
        pass