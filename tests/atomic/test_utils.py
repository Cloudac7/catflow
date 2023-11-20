import numpy as np
from catflow.atomic.utils import load_reader


def test_read_atomic_property(tmpdir):
    # Create a test dump file
    dump_file = tmpdir / 'test_dump_file.dump'
    with open(dump_file, 'w') as f:
        f.write(
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            "0.0 10.0\n0.0 10.0\n0.0 10.0\n"
            "ITEM: ATOMS id type x y z c_1[1] c_1[2]\n"
            "1 1 1.0 2.0 3.0 6.0 7.0\n"
            "2 2 4.0 5.0 6.0 8.0 9.0\n"
            "ITEM: TIMESTEP\n1\n"
            "ITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            "0.0 10.0\n0.0 10.0\n0.0 10.0\n"
            "ITEM: ATOMS id type x y z c_1[1] c_1[2]\n"
            "2 2 2.0 7.0 6.0 11.0 10.0\n"
            "1 1 1.0 3.0 3.0 12.0 9.0\n"
        )

    # Initialize DecompUtils and call read_atomic_property
    utils = load_reader(dump_file)
    results = utils.read_atomic_property()

    # Check the results
    assert isinstance(results, dict)
    assert len(results) == 2
    assert np.array_equal(
        results['c_1[1]'], np.array([[6., 8.], [12., 11.]])
    )
    assert np.array_equal(
        results['c_1[2]'], np.array([[7., 9.], [9., 10.]])
    )
