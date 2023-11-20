import numpy as np
from catflow.atomic.atomic_energy import atomic_ener_model_devi_atomic
from catflow.atomic.atomic_energy import atomic_ener_model_devi


def test_atomic_ener_model_devi_atomic(tmpdir):
    # Create a test dump file
    dummy_pe = []
    for i in range(4):
        dump_file = tmpdir / f'test_dump_file_{i}.dump'
        np.random.seed(114514 + i)
        dummy_pe_frame = np.random.rand(2)
        dummy_pe.append([dummy_pe_frame])
        with open(dump_file, 'w') as f:
            f.write(
                "ITEM: TIMESTEP\n0\n"
                "ITEM: NUMBER OF ATOMS\n2\n"
                "ITEM: BOX BOUNDS pp pp pp\n"
                "0.0 10.0\n0.0 10.0\n0.0 10.0\n"
                "ITEM: ATOMS id type x y z c_pe\n"
                f"1 1 1.0 2.0 3.0 {dummy_pe_frame[0]}\n"
                f"2 2 4.0 5.0 6.0 {dummy_pe_frame[1]}\n"
            )

    # Test atomic_ener_model_devi_atomic
    std_dev = atomic_ener_model_devi_atomic(
        *[tmpdir / f'test_dump_file_{i}.dump' for i in range(4)], key_name='c_pe'
    )
    assert np.allclose(std_dev, np.std(dummy_pe, axis=0))
    max_dev, min_dev, mean_dev = atomic_ener_model_devi(
        *[tmpdir / f'test_dump_file_{i}.dump' for i in range(4)], key_name='c_pe'
    )
    assert np.allclose(max_dev, np.max(np.std(dummy_pe, axis=0), axis=1))
    assert np.allclose(min_dev, np.min(np.std(dummy_pe, axis=0), axis=1))
    assert np.allclose(mean_dev, np.mean(np.std(dummy_pe, axis=0), axis=1))
