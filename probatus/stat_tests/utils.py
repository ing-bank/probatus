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