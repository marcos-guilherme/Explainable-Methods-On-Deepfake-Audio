from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import requests

from jmds_prepare.zenodo import (
    ChecksumError,
    DownloadResponseError,
    DownloadSizeError,
    RecordFile,
    download_verified,
    get_record_files,
)


REQUIRED_NAMES = (
    *(f"flac_T_a{suffix}.tar" for suffix in "abcde"),
    *(f"flac_D_a{suffix}.tar" for suffix in "abc"),
)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code=200,
        payload=None,
        chunks=(),
        headers=None,
        http_error=None,
    ):
        self.status_code = status_code
        self._payload = payload
        self._chunks = chunks
        self.headers = headers or {}
        self._http_error = http_error
        self.raise_for_status_called = False
        self.iter_content_chunk_size = None

    def raise_for_status(self):
        self.raise_for_status_called = True
        if self._http_error is not None:
            raise self._http_error

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        self.iter_content_chunk_size = chunk_size
        yield from self._chunks


class FakeSession:
    def __init__(self, response):
        self.responses = (
            list(response) if isinstance(response, (list, tuple)) else [response]
        )
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def make_record_payload():
    return {
        "files": [
            {
                "key": name,
                "size": index,
                "checksum": f"md5:{index:032x}",
                "links": {"self": f"https://files.example/{name}"},
            }
            for index, name in enumerate(REQUIRED_NAMES, start=1)
        ]
        + [
            {
                "key": "unrelated.txt",
                "size": 99,
                "checksum": f"md5:{99:032x}",
                "links": {"self": "https://files.example/unrelated.txt"},
            }
        ]
    }


def make_record_file(content=b"complete archive"):
    return RecordFile(
        name="flac_T_aa.tar",
        size=len(content),
        checksum=hashlib.md5(content).hexdigest(),
        download_url="https://files.example/flac_T_aa.tar",
    )


def test_get_record_files_parses_only_required_archives():
    response = FakeResponse(payload=make_record_payload())
    session = FakeSession(response)

    files = get_record_files(14498691, session=session)

    assert tuple(files) == REQUIRED_NAMES
    assert files["flac_T_aa.tar"] == RecordFile(
        name="flac_T_aa.tar",
        size=1,
        checksum=f"{1:032x}",
        download_url="https://files.example/flac_T_aa.tar",
    )
    assert session.calls == [
        ("https://zenodo.org/api/records/14498691", {"timeout": 30})
    ]
    assert response.raise_for_status_called


def test_get_record_files_rejects_missing_required_archive():
    payload = make_record_payload()
    payload["files"] = [
        item for item in payload["files"] if item["key"] != "flac_D_ac.tar"
    ]

    with pytest.raises(ValueError, match="flac_D_ac.tar"):
        get_record_files(14498691, session=FakeSession(FakeResponse(payload=payload)))


def test_download_resumes_partial_file_with_range(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    partial = tmp_path / f"{record_file.name}.partial"
    partial.write_bytes(content[:8])
    response = FakeResponse(
        status_code=206,
        chunks=(content[8:],),
        headers={"Content-Range": f"bytes 8-{len(content) - 1}/{len(content)}"},
    )
    session = FakeSession(response)

    result = download_verified(record_file, tmp_path, session=session)

    assert result == tmp_path / record_file.name
    assert result.read_bytes() == content
    assert not partial.exists()
    assert session.calls == [
        (
            record_file.download_url,
            {
                "headers": {"Range": "bytes=8-"},
                "stream": True,
                "timeout": 30,
            },
        )
    ]
    assert response.raise_for_status_called
    assert response.iter_content_chunk_size == 8 * 1024 * 1024


def test_download_reuses_valid_final_archive_without_network(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    final = tmp_path / record_file.name
    final.write_bytes(content)
    stale_partial = tmp_path / f"{record_file.name}.partial"
    stale_partial.write_bytes(b"obsolete")
    session = FakeSession(FakeResponse(status_code=500))

    result = download_verified(record_file, tmp_path, session=session)

    assert result == final
    assert result.read_bytes() == content
    assert not stale_partial.exists()
    assert session.calls == []


def test_download_quarantines_invalid_final_before_fresh_download(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    final = tmp_path / record_file.name
    final.write_bytes(b"invalid")
    session = FakeSession(FakeResponse(status_code=200, chunks=(content,)))

    result = download_verified(record_file, tmp_path, session=session)

    quarantined = list(tmp_path.glob(f"{record_file.name}.invalid*"))
    assert result.read_bytes() == content
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"invalid"
    assert session.calls


def test_download_restarts_when_server_ignores_range(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    partial = tmp_path / f"{record_file.name}.partial"
    partial.write_bytes(b"partial ")
    response = FakeResponse(status_code=200, chunks=(content,))

    result = download_verified(
        record_file, tmp_path, session=FakeSession(response)
    )

    assert result.read_bytes() == content


def test_download_rejects_unexpected_success_status_before_writing(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    partial = tmp_path / f"{record_file.name}.partial"
    partial.write_bytes(b"partial ")

    with pytest.raises(DownloadResponseError, match="204"):
        download_verified(
            record_file,
            tmp_path,
            session=FakeSession(FakeResponse(status_code=204, chunks=(content,))),
        )

    assert partial.read_bytes() == b"partial "


def test_download_preserves_partial_when_http_status_is_error(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    partial = tmp_path / f"{record_file.name}.partial"
    partial.write_bytes(b"partial ")
    response = FakeResponse(
        status_code=500,
        chunks=(content,),
        http_error=requests.HTTPError("500 Server Error"),
    )

    with pytest.raises(requests.HTTPError, match="500"):
        download_verified(
            record_file,
            tmp_path,
            session=FakeSession(response),
            max_attempts=1,
        )

    assert response.raise_for_status_called
    assert partial.read_bytes() == b"partial "


def test_download_rejects_wrong_content_range_before_appending(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    partial = tmp_path / f"{record_file.name}.partial"
    partial.write_bytes(content[:8])
    response = FakeResponse(
        status_code=206,
        chunks=(content[8:],),
        headers={"Content-Range": f"bytes 7-{len(content) - 1}/{len(content)}"},
    )

    with pytest.raises(DownloadResponseError, match="Content-Range"):
        download_verified(
            record_file,
            tmp_path,
            session=FakeSession(response),
        )

    assert partial.read_bytes() == content[:8]


@pytest.mark.parametrize(
    "content_range",
    (
        "bytes 8-15/999",
        "bytes 8-16/16",
        "bytes 8-7/16",
    ),
)
def test_download_rejects_content_range_inconsistent_with_record(
    tmp_path, content_range
):
    content = b"complete archive"
    record_file = make_record_file(content)
    partial = tmp_path / f"{record_file.name}.partial"
    partial.write_bytes(content[:8])
    response = FakeResponse(
        status_code=206,
        chunks=(content[8:],),
        headers={"Content-Range": content_range},
    )

    with pytest.raises(DownloadResponseError, match="Content-Range"):
        download_verified(
            record_file,
            tmp_path,
            session=FakeSession(response),
        )

    assert partial.read_bytes() == content[:8]


def test_download_quarantines_oversized_partial_before_fresh_download(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    partial = tmp_path / f"{record_file.name}.partial"
    partial.write_bytes(content + b"trailing corruption")
    session = FakeSession(FakeResponse(status_code=200, chunks=(content,)))

    result = download_verified(record_file, tmp_path, session=session)

    quarantined = list(tmp_path.glob(f"{record_file.name}.partial.invalid*"))
    assert result.read_bytes() == content
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == content + b"trailing corruption"
    assert session.calls[0][1]["headers"] == {}


def test_download_stops_before_stream_can_exceed_expected_size(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    partial = tmp_path / f"{record_file.name}.partial"

    with pytest.raises(DownloadSizeError, match="exceed"):
        download_verified(
            record_file,
            tmp_path,
            session=FakeSession(
                FakeResponse(status_code=200, chunks=(content + b"extra",))
            ),
        )

    assert partial.stat().st_size <= record_file.size


def test_download_retries_stream_failure_from_preserved_offset(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)

    first = FakeResponse(
        status_code=200,
        chunks=(
            content[:8],
            requests.exceptions.ChunkedEncodingError("connection lost"),
        ),
    )

    def failing_chunks(chunk_size):
        first.iter_content_chunk_size = chunk_size
        yield content[:8]
        raise requests.exceptions.ChunkedEncodingError("connection lost")

    first.iter_content = failing_chunks
    second = FakeResponse(
        status_code=206,
        chunks=(content[8:],),
        headers={"Content-Range": f"bytes 8-{len(content) - 1}/{len(content)}"},
    )
    session = FakeSession([first, second])

    result = download_verified(
        record_file, tmp_path, session=session, retry_delay=0
    )

    assert result.read_bytes() == content
    assert session.calls[1][1]["headers"] == {"Range": "bytes=8-"}


def test_download_rejects_size_mismatch_and_preserves_partial(tmp_path):
    content = b"complete archive"
    record_file = replace(make_record_file(content), size=len(content) + 1)
    partial = tmp_path / f"{record_file.name}.partial"

    with pytest.raises(DownloadSizeError, match="size"):
        download_verified(
            record_file,
            tmp_path,
            session=FakeSession(FakeResponse(chunks=(content,))),
        )

    assert partial.read_bytes() == content
    assert not (tmp_path / record_file.name).exists()


def test_download_accepts_matching_md5_and_atomically_renames(tmp_path):
    content = b"complete archive"
    record_file = make_record_file(content)
    partial = tmp_path / f"{record_file.name}.partial"

    result = download_verified(
        record_file,
        tmp_path,
        session=FakeSession(FakeResponse(chunks=(b"complete ", b"archive"))),
    )

    assert result == tmp_path / record_file.name
    assert result.read_bytes() == content
    assert not partial.exists()


def test_download_rejects_md5_mismatch_and_preserves_partial(tmp_path):
    content = b"complete archive"
    record_file = replace(make_record_file(content), checksum="0" * 32)
    partial = tmp_path / f"{record_file.name}.partial"

    with pytest.raises(ChecksumError, match="MD5"):
        download_verified(
            record_file,
            tmp_path,
            session=FakeSession(FakeResponse(chunks=(content,))),
        )

    assert partial.read_bytes() == content
    assert not (tmp_path / record_file.name).exists()
