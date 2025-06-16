"""This script checks python files for bare '# pyright: ignore' comments
and '# type: ignore' comments that are actual code comments (not in strings/docstrings).
Annotated comments like '# pyright: ignore[code]' are ignored by this script.
"""

import dataclasses
import re
import sys
import tokenize

# Regex to determine if a pyright ignore tag (the part after '# pyright: ignore')
# represents a "specific" ignore, i.e., contains one or more error codes
# enclosed in square brackets, e.g., "[code]", "[code1, code2]".
# A valid code is defined as \w[\w-]*, allowing for alphanumeric characters,
# underscores, and hyphens (e.g., 'code_1' or 'report-Unknown').
SPECIFIC_IGNORE_TAG_PATTERN = re.compile(r"^\s*\[\s*\w[\w-]*(\s*,\s*\w[\w-]*)*\s*\]$")
COMMENT_PREFIX_PATTERN = re.compile(r"^#\s*(pyright|type):\s*ignore(?:\s|$)")


@dataclasses.dataclass
class CustomLinterError:
    """Represents a specific type of linter error.

    Attributes
    ----------
    code : str
        A short code identifying the error type (e.g., "BARE_PYRIGHT_IGNORE").
    message : str
        A human-readable message describing the error.
    fix : str
        A suggestion on how to fix the error.
    """

    code: str
    message: str
    fix: str


BARE_PYRIGHT_IGNORE_ERROR = CustomLinterError(
    code="BARE_PYRIGHT_IGNORE",
    message="Bare '# pyright: ignore' comments are not allowed.",
    fix="Use '# pyright: ignore[<category>]' instead.",
)

MALFORMED_PYRIGHT_IGNORE_ERROR = CustomLinterError(
    code="MALFORMED_PYRIGHT_IGNORE",
    message="Malformed '# pyright: ignore' comment.",
    fix="Ensure the comment is in the format '# pyright: ignore[<category>]'.",
)

TYPE_IGNORE_ERROR = CustomLinterError(
    code="TYPE_IGNORE",
    message='"# type: ignore" comments are not allowed.',
    fix='Use "# pyright: ignore[<category>]" instead.',
)


def main() -> None:  # noqa: C901
    """Check Python files for disallowed linter directive comments.

    Iterates through Python files specified as command-line arguments.
    For each file, it tokenizes the content and inspects comments to
    identify and report:
    - Bare '# pyright: ignore' comments.
    - Malformed '# pyright: ignore[<category>]' comments.
    - '# type: ignore' comments.

    Error messages, including the file path, line number, and offending line,
    are printed to standard output for each violation.

    The script exits with a status code of 1 if any disallowed comments
    or processing errors are found, and 0 otherwise.

    Raises
    ------
    SystemExit
        If disallowed comments or processing errors are encountered, exits with status 1.
        Otherwise, exits with status 0.
    """
    found_linter_error = False

    for filename in sys.argv[1:]:
        try:
            # Read file content for line extraction first
            with open(filename, encoding="utf-8") as f_text:
                file_content = f_text.read()
            lines = file_content.splitlines()

            # Tokenize requires opening in binary mode
            with open(filename) as f_bin:
                tokens = tokenize.generate_tokens(f_bin.readline)

                for token_info in tokens:
                    if token_info.type == tokenize.COMMENT:
                        comment_text = token_info.string
                        line_number = token_info.start[0]

                        match = COMMENT_PREFIX_PATTERN.match(comment_text)
                        if match:
                            error_to_report: CustomLinterError | None = None
                            tag_content = comment_text[match.end() :].split("#", 1)[0].strip()

                            if match.group(1) == "type":
                                error_to_report = TYPE_IGNORE_ERROR
                            elif match.group(1) == "pyright":
                                if not tag_content:  # Bare 'pyright: ignore'
                                    error_to_report = BARE_PYRIGHT_IGNORE_ERROR
                                elif not SPECIFIC_IGNORE_TAG_PATTERN.fullmatch(tag_content):
                                    error_to_report = MALFORMED_PYRIGHT_IGNORE_ERROR

                            if error_to_report:
                                reason = f"{error_to_report.message} {error_to_report.fix}"
                                print(f"{filename}:{line_number}: {reason}\n\t{lines[line_number - 1].strip()}")
                                found_linter_error = True
        except (  # noqa: PERF203
            tokenize.TokenError,
            SyntaxError,
            UnicodeDecodeError,
        ) as e:  # Catch tokenizing, syntax, and file encoding errors
            print(f"Error parsing/tokenizing Python file {filename}: {e}")
            found_linter_error = True  # Treat parse/tokenize errors as failures
        except Exception as e:  # noqa: BLE001
            print(f"Error processing file {filename}: {e}")
            found_linter_error = True  # Treat other errors as failures

    if found_linter_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
