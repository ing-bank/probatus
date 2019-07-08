import pyrisk.datasets as ds

def test_lending_club_shape():
    assert ds.lending_club().shape==(10429, 18)
