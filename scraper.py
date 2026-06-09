import hashlib
import io
import logging
from pathlib import Path

import httpx
from PIL import Image
from duckduckgo_search import DDGS
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageScraper:
    """Download images from DuckDuckGo image search — no browser, no API key."""

    def __init__(
        self,
        images_dir: str = "images",
        image_quality: int = 85,
        timeout: int = 10,
    ):
        self.images_dir = Path(images_dir)
        self.image_quality = image_quality
        self.timeout = timeout

    def search_and_download(self, query: str, n_images: int = 100) -> Path:
        folder = self.images_dir / "_".join(query.lower().split())
        folder.mkdir(parents=True, exist_ok=True)

        # Fetch extra results to account for download failures
        results = list(DDGS().images(query, max_results=min(n_images * 3, 500)))
        urls = [r["image"] for r in results]

        saved = 0
        with tqdm(total=n_images, desc=f"Downloading '{query}'") as pbar:
            for url in urls:
                if saved >= n_images:
                    break
                if self._save_image(folder, url):
                    saved += 1
                    pbar.update(1)

        print(f"Saved {saved}/{n_images} images → {folder}")
        return folder

    def _save_image(self, folder: Path, url: str) -> bool:
        try:
            r = httpx.get(url, timeout=self.timeout, follow_redirects=True)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            name = hashlib.sha1(r.content).hexdigest()[:12] + ".jpg"
            path = folder / name
            if not path.exists():
                img.save(path, "JPEG", quality=self.image_quality)
            return True
        except Exception as exc:
            logger.debug("Skip %s — %s", url, exc)
            return False
