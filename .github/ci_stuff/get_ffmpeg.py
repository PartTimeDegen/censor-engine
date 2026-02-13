import platform
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# NOTE: This is just from ChatGPT because I can't be asked un-implementing it

FFMPEG_DIR = Path(f"tools/ffmpeg/{platform.system()}")
FFMPEG_DIR.mkdir(parents=True, exist_ok=True)


def ffmpeg_exists() -> bool:
    for name in ["ffmpeg", "ffmpeg.exe"]:
        if (FFMPEG_DIR / name).exists():
            return True
    return False


def download(url: str, dest: Path):
    print(f"Downloading FFmpeg from {url}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")


def extract(archive: Path):
    print("Extracting...")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(FFMPEG_DIR)
    else:
        with tarfile.open(archive, "r:*") as t:
            t.extractall(FFMPEG_DIR)
    archive.unlink()
    print("Extraction complete.")


def main():
    if ffmpeg_exists():
        print("FFmpeg already present.")
        return

    system = platform.system()

    if system == "Windows":
        url = (
            "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        )
        archive = FFMPEG_DIR / "ffmpeg.zip"

    elif system == "Linux":
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        archive = FFMPEG_DIR / "ffmpeg.tar.xz"

    elif system == "Darwin":
        url = "https://evermeet.cx/ffmpeg/getrelease/zip"
        archive = FFMPEG_DIR / "ffmpeg.zip"

    else:
        raise RuntimeError("Unsupported OS")

    download(url, archive)
    extract(archive)


if __name__ == "__main__":
    main()
