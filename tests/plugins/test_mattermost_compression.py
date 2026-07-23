"""Direct unit tests for MattermostAdapter._maybe_compress_image.

These tests don't touch the network or Mattermost at all — they exercise the
compressor in isolation by constructing a real PIL image larger than the 800 KB
threshold and asserting the output is smaller, in JPEG format, and still decodes.

The function is critical for avoiding 413s on nginx-fronted Mattermost servers
with default ``client_max_body_size: 1m`` — see plugins/platforms/mattermost/adapter.py
docstring for the threshold evidence.
"""
import io
import os
import sys
from pathlib import Path

import pytest

# Add the source tree to sys.path so the test can import the adapter without
# needing the plugin loader.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plugins.platforms.mattermost.adapter import MattermostAdapter  # noqa: E402


def _make_large_png_bytes(target_kb: int = 1500) -> bytes:
    """Generate a real PNG that exceeds `target_kb` bytes.

    Uses a 1024×1024 RGB image with smooth gradients so PNG compression actually
    applies — random pixels would compress poorly and might exceed PIL's
    internal limits. The gradient pattern lands near the target on first try
    because PIL's zlib pass is predictable on uniform regions."""
    from PIL import Image

    img = Image.new("RGB", (1024, 1024))
    pixels = []
    for y in range(1024):
        for x in range(1024):
            # Smooth diagonal gradient — PNG-friendly (compresses to ~3 MB)
            r = (x * 7 + y * 3) % 256
            g = (x * 5 + y * 11) % 256
            b = (x * 13 + y * 2) % 256
            pixels.append((r, g, b))
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=0)
    return buf.getvalue()


def _make_small_png_bytes() -> bytes:
    """Generate a small PNG that's well under the 800 KB threshold."""
    from PIL import Image

    img = Image.new("RGB", (16, 16), color=(60, 60, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def adapter():
    """A MattermostAdapter instance with no live connection.

    Most attributes used by _maybe_compress_image are defaults, but we still
    need an instance to call the method."""
    from types import SimpleNamespace

    from gateway.config import PlatformConfig

    cfg = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"url": "https://mm.test"},
    )
    return MattermostAdapter(cfg)


class TestMaybeCompressImage:
    """Behavior contract for _maybe_compress_image.

    Verifies the four essential promises:
      1. Sub-800 KB images pass through untouched.
      2. Non-image content types pass through untouched.
      3. PNG/JPEG/RGBA images above 800 KB are compressed under 800 KB.
      4. PIL failures degrade gracefully (returns original bytes)."""

    @pytest.mark.asyncio
    async def test_small_png_passes_through_unchanged(self, adapter):
        """Images under 800 KB must not be re-encoded — preserves quality."""
        original = _make_small_png_bytes()
        result = await adapter._maybe_compress_image(
            original, "tiny.png", "image/png"
        )
        assert result == original, "small images should not be modified"

    @pytest.mark.asyncio
    async def test_small_jpeg_passes_through_unchanged(self, adapter):
        original = _make_small_png_bytes()  # bytes are bytes — content_type gates
        result = await adapter._maybe_compress_image(
            original, "tiny.jpg", "image/jpeg"
        )
        assert result == original

    @pytest.mark.asyncio
    async def test_non_image_content_type_passes_through(self, adapter):
        """PDFs, zips, videos etc. must not be touched even if huge."""
        # Pretend it's a 2 MB "blob" with a non-image content_type.
        original = b"X" * (2 * 1024 * 1024)
        result = await adapter._maybe_compress_image(
            original, "thing.pdf", "application/pdf"
        )
        assert result == original, "non-image content must not be touched"

    @pytest.mark.asyncio
    async def test_jpg_alias_jpeg_passes_through(self, adapter):
        """`image/jpg` is a common alias for `image/jpeg` — should be accepted."""
        original = _make_small_png_bytes()
        result = await adapter._maybe_compress_image(
            original, "thing.jpg", "image/jpg"
        )
        assert result == original

    @pytest.mark.asyncio
    async def test_oversized_rgb_png_compressed_under_threshold(self, adapter):
        """The whole point: a 1.5 MB+ PNG must come back under 800 KB."""
        original = _make_large_png_bytes(target_kb=1500)
        assert len(original) > 800 * 1024, (
            f"Test setup error: generated PNG only {len(original)} bytes, "
            "need >800 KB to exercise the compression path"
        )
        result = await adapter._maybe_compress_image(
            original, "large.png", "image/png"
        )
        assert len(result) < len(original), (
            f"compression didn't shrink: {len(original)} → {len(result)}"
        )
        assert len(result) <= 800 * 1024 + 4096, (
            f"compressed result {len(result)} bytes is over the 800 KB target "
            "(small overshoot OK due to JPEG header, but should be close)"
        )

    @pytest.mark.asyncio
    async def test_compressed_result_is_valid_jpeg(self, adapter):
        """If we say the image is RGB (not RGBA), the output should be JPEG."""
        from PIL import Image

        original = _make_large_png_bytes(target_kb=1500)
        result = await adapter._maybe_compress_image(
            original, "large.png", "image/png"
        )
        # Should be a valid JPEG image we can re-open
        img = Image.open(io.BytesIO(result))
        assert img.format == "JPEG", (
            f"expected JPEG output for RGB image, got {img.format}"
        )

    @pytest.mark.asyncio
    async def test_rgba_png_kept_as_png(self, adapter):
        """RGBA images must stay as PNG (no alpha → JPEG conversion)."""
        from PIL import Image

        # Build a 1024×1024 RGBA PNG (alpha channel forces PNG output)
        img = Image.new("RGBA", (1024, 1024))
        pixels = []
        for y in range(1024):
            for x in range(1024):
                pixels.append((x % 256, y % 256, (x + y) % 256, 128))
        img.putdata(pixels)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=0)
        original = buf.getvalue()
        assert len(original) > 800 * 1024, (
            f"RGBA setup failed: {len(original)} bytes, need >800 KB"
        )

        result = await adapter._maybe_compress_image(
            original, "rgba.png", "image/png"
        )
        out_img = Image.open(io.BytesIO(result))
        assert out_img.format == "PNG", (
            f"RGBA must stay PNG (alpha channel), got {out_img.format}"
        )
        assert out_img.mode == "RGBA", (
            f"alpha channel must be preserved, got mode={out_img.mode}"
        )

    @pytest.mark.asyncio
    async def test_pil_failure_returns_original_unchanged(self, adapter, monkeypatch):
        """If PIL raises (corrupt image, missing dep), we return original bytes.

        Delivery must not be blocked by a compression bug — the function logs
        a warning and degrades to uploading the original."""
        # Monkey-patch Image.open to raise — simulates a corrupt PNG.
        from PIL import Image as PILImage

        def broken_open(*_args, **_kwargs):
            raise OSError("simulated PIL failure")

        monkeypatch.setattr(PILImage, "open", broken_open)

        original = _make_large_png_bytes(target_kb=1500)
        result = await adapter._maybe_compress_image(
            original, "broken.png", "image/png"
        )
        assert result == original, (
            "PIL failure must not corrupt the upload — return original bytes"
        )