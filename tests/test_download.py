import io
import zipfile

import pytest

from thinedge.download import extract_json_members, source_url


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)


def test_source_url():
    assert source_url("ipl") == "https://cricsheet.org/downloads/ipl_json.zip"


@pytest.mark.parametrize("bad", ["", "../etc", "ipl/../x", "IPL", "ipl.zip", "a b"])
def test_source_name_validated(bad):
    with pytest.raises(ValueError):
        source_url(bad)


def test_extract_only_safe_flat_json(tmp_path):
    archive = tmp_path / "a.zip"
    make_zip(
        archive,
        {
            "1001.json": "{}",
            "README.txt": "hello",
            "../evil.json": "{}",
            "sub/1002.json": "{}",
            "..\\evil2.json": "{}",
            "weird name.json": "{}",
        },
    )
    dest = tmp_path / "out"
    dest.mkdir()
    assert extract_json_members(archive, dest) == 1
    assert sorted(p.name for p in dest.iterdir()) == ["1001.json"]
    assert not (tmp_path / "evil.json").exists()


def test_extract_rejects_oversized_member(tmp_path, monkeypatch):
    monkeypatch.setattr("thinedge.download.MAX_MEMBER_BYTES", 10)
    archive = tmp_path / "a.zip"
    make_zip(archive, {"1.json": "x" * 100})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ValueError):
        extract_json_members(archive, dest)
