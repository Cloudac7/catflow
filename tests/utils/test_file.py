from catflow.utils.file import tail
from catflow.utils.file import count_lines

def test_tail(shared_datadir, tmp_path):
    with open(shared_datadir / "test_two_frames.xyz", 'rb') as f:
        last = tail(f, lines=26)
    with open(tmp_path / "test_last.xyz", 'wb') as output:
        output.write(last)
    with open(tmp_path / "test_last.xyz") as f:
        file_tailed = f.readlines()
    with open(shared_datadir / "test_last_frame.xyz") as f:
        file_ref = f.readlines()
    
    assert file_tailed == file_ref

def test_count_lines(shared_datadir):
    with open(shared_datadir / "test_last_frame.xyz") as f:
        count = count_lines(f)
    assert count == 26
