from pyrisk.datasets import lending_club

def test_lending_club_shape():
    assert lending_club().shape==(10429, 18)
