"""
TODO: DOCSTRING
"""

class BaseDependencePlotter:
    """
    TODO: DOCSTRING
    """
    def __init__(self, )
        pass

    
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier 
    
    X, y = make_classification(
        n_samples = 1000,
        n_features = 8,
        n_informative = 3
    )
    
    clf = RandomForestClassifier()
    
    clf.fit(X, y)