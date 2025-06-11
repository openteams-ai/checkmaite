import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BASE_DIR / "scripts" / "check_bare_pyright_ignore.py"

# Expected error messages based on CustomLinterError definitions in the script
TYPE_IGNORE_MSG = '"# type: ignore" comments are not allowed. Use "# pyright: ignore[<category>]" instead.'
BARE_PYRIGHT_IGNORE_MSG = (
    "Bare '# pyright: ignore' comments are not allowed. Use '# pyright: ignore[<category>]' instead."
)
MALFORMED_PYRIGHT_IGNORE_MSG = (
    "Malformed '# pyright: ignore' comment. Ensure the comment is in the format '# pyright: ignore[<category>]'."
)


def run_script_under_test(file_paths):
    """Helper method to run the script with given file paths."""
    cmd = [sys.executable, str(SCRIPT_PATH)] + file_paths
    # Use encoding="utf-8" to decode stdout/stderr consistently
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")  # noqa: S603


@pytest.fixture
def temp_file_creator():
    """Fixture to create temporary files and ensure their cleanup."""
    created_files = []

    def _create(content, suffix=".py", mode="w", encoding="utf-8"):
        # The file needs to persist after close() for the script to read it.
        tmp_file = tempfile.NamedTemporaryFile(mode=mode, delete=False, suffix=suffix, encoding=encoding)
        tmp_file.write(content)
        tmp_file.close()
        created_files.append(tmp_file.name)
        return tmp_file.name

    yield _create
    for f_path_str in created_files:
        # Check existence before removing
        f_path = Path(f_path_str)
        if f_path.exists():
            f_path.unlink()


def test_no_files_provided():
    """Test script exits successfully when no files are provided."""
    process = run_script_under_test([])
    assert process.returncode == 0, "Script should exit 0 for no files."
    assert process.stdout == "", "Stdout should be empty for no files."
    assert process.stderr == "", "Stderr should be empty for no files."


@pytest.mark.parametrize(
    ("content_description", "file_content"),
    [
        ("single_error_code", "import os\nprint('hello')  # pyright: ignore[reportUnknownMemberType]\n"),
        ("multiple_error_codes", "import os\nprint('hello')  # pyright: ignore[reportUnknownMemberType, call-arg]\n"),
        ("single_error_with_space", "import os\nprint('hello')  # pyright: ignore [misc]\n"),
        (
            "multiple_errors_with_space",
            "import os\nprint('hello')  # pyright: ignore [reportUnknownMemberType, call-arg]\n",
        ),
        (
            "mixed_valid_ignores",
            "import os\nprint('hello')  # pyright: ignore[reportUnknownMemberType]\nprint('world')  # pyright: ignore [misc]\n",
        ),
        ("no_space_multiple", "import os\nprint('hello')  # pyright: ignore[reportUnknownMemberType,call-arg]\n"),
        (
            "complex_error_codes",
            "import os\nprint('hello')  # pyright: ignore[reportUnknownMemberType, call-arg, type-arg]\n",
        ),
        ("bare_ignore_in_single_quote_string", "s = '# pyright: ignore'\nprint(s)\n"),
        ("bare_ignore_in_double_quote_string", 's = "# pyright: ignore"\nprint(s)\n'),
        ("bare_ignore_in_triple_quote_docstring", '"""\n# pyright: ignore\n"""\npass\n'),
        ("bare_ignore_in_triple_quote_multiline_string", "s = '''\n# pyright: ignore\n'''\nprint(s)\n"),
        ("specific_ignore_in_string", "s = '# pyright: ignore[code]'\nprint(s)\n"),
        ("commented_out_bare_ignore", "# print('hello') # pyright: ignore\n"),
        ("commented_out_specific_ignore", "# print('hello') # pyright: ignore[code]\n"),
    ],
)
def test_file_no_bare_ignore(temp_file_creator, content_description, file_content):
    """Test script with files containing valid (non-bare) type ignores."""
    tmpfile_path = temp_file_creator(file_content)
    process = run_script_under_test([tmpfile_path])
    assert process.returncode == 0, f"Script should exit 0 for {content_description}. Stdout: {process.stdout}"
    assert process.stdout == "", f"Stdout should be empty for {content_description}. Got: {process.stdout}"


@pytest.mark.parametrize(
    (
        "content_description",
        "file_content",
        "expected_error_line_no",
        "expected_error_line_stripped_content",
        "expected_reason_msg",
    ),
    [
        (
            "bare_ignore_comment_suffix",  # tag is "- then more text" -> MALFORMED
            "print('problem')  # pyright: ignore - then more text\n",
            1,
            "print('problem')  # pyright: ignore - then more text",
            MALFORMED_PYRIGHT_IGNORE_MSG,
        ),
        (
            "bare_ignore_trailing_spaces",  # tag is "" -> BARE
            "print('problem')  # pyright: ignore  \n",
            1,
            "print('problem')  # pyright: ignore",
            BARE_PYRIGHT_IGNORE_MSG,
        ),
        (
            "bare_ignore_trailing_tab",  # tag is "" -> BARE
            "print('problem')  # pyright: ignore\t\n",
            1,
            "print('problem')  # pyright: ignore",
            BARE_PYRIGHT_IGNORE_MSG,
        ),
        (
            "specific_ignore_incomplete_bracket",  # tag is "[" -> MALFORMED
            "print('problem')  # pyright: ignore [",
            1,
            "print('problem')  # pyright: ignore [",
            MALFORMED_PYRIGHT_IGNORE_MSG,
        ),
        (
            "specific_ignore_empty_brackets",  # tag is "[]" -> MALFORMED
            "print('problem')  # pyright: ignore []",
            1,
            "print('problem')  # pyright: ignore []",
            MALFORMED_PYRIGHT_IGNORE_MSG,
        ),
        (
            "specific_ignore_empty_brackets_with_space",  # tag is "[ ]" -> MALFORMED
            "print('problem')  # pyright: ignore [ ]",
            1,
            "print('problem')  # pyright: ignore [ ]",
            MALFORMED_PYRIGHT_IGNORE_MSG,
        ),
        (
            "string_with_ignore_then_real_bare_ignore_comment",  # tag is "" -> BARE
            """s = "text with # pyright: ignore inside"
            print(s) # pyright: ignore""",
            2,
            "print(s) # pyright: ignore",
            BARE_PYRIGHT_IGNORE_MSG,
        ),
        (
            "bare_ignore_with_secondary_comment",  # tag is "- reason" -> MALFORMED
            "print('problem') # pyright: ignore - reason # secondary comment",
            1,
            "print('problem') # pyright: ignore - reason # secondary comment",
            MALFORMED_PYRIGHT_IGNORE_MSG,
        ),
    ],
)
def test_file_with_various_bare_ignores(
    temp_file_creator,
    content_description,
    file_content,
    expected_error_line_no,
    expected_error_line_stripped_content,
    expected_reason_msg,
):
    """Test script with various forms of 'pyright: ignore' comments that should fail."""
    tmpfile_path = temp_file_creator(file_content)
    process = run_script_under_test([tmpfile_path])

    expected_returncode = 1  # All these cases should fail

    assert (
        process.returncode == expected_returncode
    ), f"Failed for {content_description}: return code mismatch. Expected {expected_returncode}, got {process.returncode}. stdout: '{process.stdout}', stderr: '{process.stderr}'"

    expected_line_output = (
        f"{tmpfile_path}:{expected_error_line_no}: {expected_reason_msg}\n\t{expected_error_line_stripped_content}"
    )
    # Use strip() on process.stdout for comparison to handle potential trailing newlines from the script's print
    assert (
        expected_line_output.strip() in process.stdout.strip()
    ), f"Failed for {content_description}: stdout mismatch. Expected '{expected_line_output.strip()}' in '{process.stdout.strip()}'"


def test_file_with_type_ignore_comment(temp_file_creator):
    """Test script with a file containing a '# type: ignore' comment."""
    file_content = "problem_line = 1  # type: ignore\n"
    stripped_line_content = "problem_line = 1  # type: ignore"
    tmpfile_path = temp_file_creator(file_content)
    process = run_script_under_test([tmpfile_path])

    assert process.returncode == 1, (
        "Script should exit 1 for '# type: ignore' comment. " f"Stdout: '{process.stdout}', Stderr: '{process.stderr}'"
    )

    # Construct the expected stdout message precisely using TYPE_IGNORE_MSG.
    expected_stdout_message = f"{tmpfile_path}:1: {TYPE_IGNORE_MSG}\n\t{stripped_line_content}"

    # process.stdout might have a trailing newline depending on the system/subprocess behavior.
    # Comparing stripped versions handles this.
    assert process.stdout.strip() == expected_stdout_message, (
        f"Stdout mismatch for '# type: ignore' case.\n"
        f"Expected:\n'{expected_stdout_message}'\n"
        f"Got:\n'{process.stdout.strip()}'"
    )


def test_empty_file(temp_file_creator):
    """Test script with an empty file."""
    tmpfile_path = temp_file_creator("")
    process = run_script_under_test([tmpfile_path])
    assert process.returncode == 0
    assert process.stdout == ""


def test_non_existent_file():
    """Test script with a non-existent file path."""
    non_existent_path = "some_file_that_does_not_exist.py"
    process = run_script_under_test([non_existent_path])
    assert process.returncode == 1
    assert f"Error processing file {non_existent_path}:" in process.stdout


def test_file_with_non_utf8_encoding(temp_file_creator):
    """Test script with a file that is not UTF-8 encoded."""
    # Create a file with Latin-1 encoding that will fail UTF-8 decoding
    content_latin1 = "print('olé')".encode("latin-1")
    tmpfile_path = temp_file_creator(content_latin1, mode="wb", encoding=None)  # Write bytes
    process = run_script_under_test([tmpfile_path])
    assert process.returncode == 1, f"Script should exit 1 for encoding error. Stdout: {process.stdout}"
    assert f"Error parsing/tokenizing Python file {tmpfile_path}:" in process.stdout
    # Check for part of the typical UnicodeDecodeError message
    assert "decode byte" in process.stdout.lower()


def test_multiple_files_no_issues(temp_file_creator):
    """Test with multiple files, none having bare ignores."""
    content1 = "print(1) # pyright: ignore[call-arg]\n"
    content2 = "print(2) # pyright: ignore [name-defined]\n"
    tmpfile1_path = temp_file_creator(content1)
    tmpfile2_path = temp_file_creator(content2)
    process = run_script_under_test([tmpfile1_path, tmpfile2_path])
    assert process.returncode == 0
    assert process.stdout == ""


def test_multiple_files_one_bare_ignore(temp_file_creator):
    """Test with multiple files, one having a bare ignore."""
    content1 = "print(1) # pyright: ignore[call-arg]\n"
    content2_bare = "print(2) # pyright: ignore\n"
    tmpfile1_path = temp_file_creator(content1)
    tmpfile2_path = temp_file_creator(content2_bare)
    process = run_script_under_test([tmpfile1_path, tmpfile2_path])
    assert process.returncode == 1
    # content2_bare is a bare pyright: ignore
    expected_line = f"{tmpfile2_path}:1: {BARE_PYRIGHT_IGNORE_MSG}\n\t{content2_bare.strip()}"
    assert expected_line.strip() in process.stdout.strip()
    assert tmpfile1_path not in process.stdout  # Ensure clean file is not reported


def test_multiple_files_all_bare_ignores(temp_file_creator):
    """Test with multiple files, all having bare ignores."""
    content1_bare = "a = 1  # pyright: ignore\n"
    content2_bare = "b = 2  # pyright: ignore\n"
    tmpfile1_path = temp_file_creator(content1_bare)
    tmpfile2_path = temp_file_creator(content2_bare)
    process = run_script_under_test([tmpfile1_path, tmpfile2_path])
    assert process.returncode == 1
    # Both are bare pyright: ignore comments
    expected_line1 = f"{tmpfile1_path}:1: {BARE_PYRIGHT_IGNORE_MSG}\n\t{content1_bare.strip()}"
    expected_line2 = f"{tmpfile2_path}:1: {BARE_PYRIGHT_IGNORE_MSG}\n\t{content2_bare.strip()}"
    # Check if each expected line (stripped) is present in the stripped stdout
    # This handles cases where multiple errors are printed in any order, each potentially with a trailing newline
    stdout_stripped = process.stdout.strip()
    assert expected_line1.strip() in stdout_stripped
    assert expected_line2.strip() in stdout_stripped
