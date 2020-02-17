from probatus.datasets import lending_club

def test_lending_club_shape():
    assert lending_club()[0].shape==(10429, 18)
