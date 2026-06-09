import pytest
from pathlib import Path
from unittest.mock import patch
from autovision.scraper import ImageScraper


def _patched_scrape(scraper, query, n_images, ddgs_results=None):
    """Helper: run search_and_download with DDGS mocked out."""
    with patch("autovision.scraper.DDGS") as mock_ddgs:
        mock_ddgs.return_value.images.return_value = ddgs_results or []
        scraper.search_and_download(query, n_images=n_images)


def test_creates_folder_for_query(tmp_path):
    scraper = ImageScraper(images_dir=str(tmp_path))
    _patched_scrape(scraper, "golden retriever", n_images=5)
    assert (tmp_path / "golden_retriever").is_dir()


def test_spaces_in_query_become_underscores(tmp_path):
    scraper = ImageScraper(images_dir=str(tmp_path))
    _patched_scrape(scraper, "siberian husky puppy", n_images=1)
    assert (tmp_path / "siberian_husky_puppy").is_dir()


def test_failed_download_does_not_crash(tmp_path):
    """A bad URL (httpx raises) should be silently skipped, not raise."""
    scraper = ImageScraper(images_dir=str(tmp_path))
    with patch("autovision.scraper.DDGS") as mock_ddgs:
        mock_ddgs.return_value.images.return_value = [
            {"image": "https://not-real.invalid/img.jpg"},
        ]
        with patch("autovision.scraper.httpx.get", side_effect=Exception("timeout")):
            scraper.search_and_download("test", n_images=3)
    # folder should still have been created
    assert (tmp_path / "test").is_dir()


def test_no_results_saves_zero_images(tmp_path):
    scraper = ImageScraper(images_dir=str(tmp_path))
    _patched_scrape(scraper, "obscure query", n_images=10, ddgs_results=[])
    folder = tmp_path / "obscure_query"
    assert folder.is_dir()
    assert list(folder.iterdir()) == []
