from pathlib import Path
from tempfile import TemporaryDirectory

from checkmaite.core.report._markdown import create_markdown_output


def _write_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # write some bytes that are valid PNG header-ish (not a real image but filesystem check is fine)
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")


def test_create_markdown_output_copies_local_image_and_rewrites_md():
    with TemporaryDirectory() as tmp_img_dir, TemporaryDirectory() as tmp_out_dir:
        img_dir = Path(tmp_img_dir)
        out_dir = Path(tmp_out_dir)

        src_img = img_dir / "figure1.png"
        _write_dummy_image(src_img)

        md = f"![Figure]({src_img.as_posix()})\n"

        create_markdown_output(md, out_dir, md_filename="report.md")

        # Saved markdown and copied image should exist
        saved_md = out_dir / "report.md"
        assert saved_md.exists()

        copied_img = out_dir / "images" / src_img.name
        assert copied_img.exists()

        saved_text = saved_md.read_text(encoding="utf-8")
        # The saved markdown should reference the images/ path
        assert f"(images/{src_img.name})" in saved_text

        # Display handled; saved file assertions above validate behavior


def test_create_markdown_output_skips_remote_urls_and_leaves_md_unchanged():
    with TemporaryDirectory() as tmp_out_dir:
        out_dir = Path(tmp_out_dir)
        remote_url = "https://example.com/image.png"
        md = f"![Remote]({remote_url})\n"

        create_markdown_output(md, out_dir, md_filename="report.md")

        saved_md = out_dir / "report.md"
        assert saved_md.exists()
        saved_text = saved_md.read_text(encoding="utf-8")

        # remote URL should remain unchanged in the saved markdown
        assert f"({remote_url})" in saved_text
        # no images copied
        images_dir = out_dir / "images"
        assert images_dir.exists()
        assert len(list(images_dir.iterdir())) == 0


def test_create_markdown_output_handles_file_scheme_paths():
    with TemporaryDirectory() as tmp_img_dir, TemporaryDirectory() as tmp_out_dir:
        img_dir = Path(tmp_img_dir)
        out_dir = Path(tmp_out_dir)

        src_img = img_dir / "pic.png"
        _write_dummy_image(src_img)

        file_uri = f"file://{src_img.as_posix()}"
        md = f"![FileURI]({file_uri})\n"

        create_markdown_output(md, out_dir, md_filename="report.md")

        saved_md = out_dir / "report.md"
        txt = saved_md.read_text(encoding="utf-8")

        # file:// URLs are not rewritten or copied
        assert file_uri in txt
        assert not (out_dir / "images" / src_img.name).exists()


def test_data_uri_is_skipped_and_left_unchanged():
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"

    with TemporaryDirectory() as tmp_out_dir:
        out_dir = Path(tmp_out_dir)
        md = f"![Data]({data_uri})\n"

        create_markdown_output(md, out_dir, md_filename="report.md")

        saved = out_dir / "report.md"
        text = saved.read_text(encoding="utf-8")

        assert data_uri in text
        assert len(list((out_dir / "images").iterdir())) == 0


def test_create_markdown_output_relative_and_duplicate_names():
    with TemporaryDirectory() as tmp_img_dir, TemporaryDirectory() as tmp_out_dir:
        img_dir = Path(tmp_img_dir)
        out_dir = Path(tmp_out_dir)

        # prepare an existing images dir with a file that will cause a name collision
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        existing = images_dir / "dup.png"
        existing.write_bytes(b"existing")

        # create source image with same name
        src = img_dir / "dup.png"
        _write_dummy_image(src)

        md = f"![Dup]({src.as_posix()})\n"
        create_markdown_output(md, out_dir, md_filename="report.md")

        saved_md = out_dir / "report.md"
        assert saved_md.exists()
        txt = saved_md.read_text(encoding="utf-8")

        # Because an existing file with same name was present, the new file should be renamed
        assert "images/dup.png" not in txt
        # should reference a renamed file like dup_1.png
        assert "images/dup_" in txt


def test_create_markdown_output_uses_cwd_fallback_for_relative_paths(tmp_path, monkeypatch):
    # create an image file in the current working dir
    img_name = "local_cwd.png"
    cwd_img = Path.cwd() / img_name
    try:
        cwd_img.write_bytes(b"\x89PNG\r\n\x1a\n")

        md = f"![Local]({img_name})\n"
        with TemporaryDirectory() as tmp_out:
            out_dir = Path(tmp_out)
            create_markdown_output(md, out_dir, md_filename="report.md")

            saved_md = out_dir / "report.md"
            assert saved_md.exists()
            txt = saved_md.read_text(encoding="utf-8")
            # should have copied the image from cwd into out_dir/images
            assert "(images/local_cwd.png)" in txt
            assert (out_dir / "images" / "local_cwd.png").exists()

    finally:
        try:
            cwd_img.unlink()
        except FileNotFoundError:
            # file already removed or never created; ignore
            pass
