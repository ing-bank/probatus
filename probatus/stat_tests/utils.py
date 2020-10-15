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


import functools

def verbose_p_vals(func):
    """Decorator to enable verbose printing of p-vals"""

    @functools.wraps(func)
    def wrapper_verbose_p_vals(*args, **kwargs):
        test_name = func.__name__.upper()

        stat, pvalue = func(*args, **kwargs)

        if "verbose" in kwargs and kwargs["verbose"] is True:
            print("\n{}: pvalue =".format(test_name), pvalue)
            if pvalue < 0.01:
                print(
                    "\n{}: Null hypothesis rejected with 99% confidence. Distributions very different.".format(
                        test_name
                    )
                )
            elif pvalue < 0.05:
                print(
                    "\n{}: Null hypothesis rejected with 95% confidence. Distributions different.".format(
                        test_name
                    )
                )
            else:
                print(
                    "\n{}: Null hypothesis cannot be rejected. Distributions not statistically different.".format(
                        test_name
                    )
                )

        return stat, pvalue

    return wrapper_verbose_p_vals