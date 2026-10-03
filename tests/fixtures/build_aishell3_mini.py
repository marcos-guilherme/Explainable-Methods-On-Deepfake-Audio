"""Build tests/fixtures/aishell3_mini.tgz — run once, commit the archive."""

import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "aishell3_mini.tgz"

MEMBERS = {
    "spk-info.txt": (
        "# voice-file name; age group; gender; accent\n"
        "SSB0005\tB\tfemale\tnorth\n"
        "SSB0999\tB\tmale\tsouth\n"
    ),
    "train/content.txt": (
        "SSB00050001.wav\t你好 ni3 hao3\n"
        "SSB00050002.wav\t训练语句 xun4 lian4\n"
        "SSB09990001.wav\t无wav训练\n"
    ),
    "test/content.txt": (
        "SSB00050003.wav\t测试一 ce4 shi4 yi1\n"
        "SSB00050004.wav\t测试二 ce4 shi4 er4\n"
        "SSB00050005.wav\t缺wav测试\n"
    ),
    "train/label_train-set.txt": (
        "# Prosody labeling instructions\n"
        "SSB00050001|ni3 hao3\n"
        "SSB00050002|xun4 lian4\n"
    ),
}

WAV_STUBS = {
    "train/wav/SSB0005/SSB00050001.wav": b"RIFFxxxxWAVEfmt ",
    "train/wav/SSB0005/SSB00050002.wav": b"RIFFyyyyWAVEfmt ",
    "test/wav/SSB0005/SSB00050003.wav": b"RIFFzzzzWAVEfmt ",
    "test/wav/SSB0005/SSB00050004.wav": b"RIFFwwwwWAVEfmt ",
    "test/wav/SSB0005/SSB00059999.wav": b"RIFFextraWAVEfmt ",
}


def build() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for name, text in MEMBERS.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        for name, payload in WAV_STUBS.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":
    build()
    print(f"wrote {ARCHIVE}")
