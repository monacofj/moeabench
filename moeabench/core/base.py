# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import re

class _SilentDisplayedText(str):
    """String payload that stays quiet when it is the last notebook expression."""

    def __repr__(self):
        return ""

    def _repr_pretty_(self, p, cycle):
        p.text("")

    def _repr_markdown_(self):
        return ""


class Reportable:
    """
    Mixin for objects that support narrative reporting in MoeaBench.
    Provides a consistent interface for environment-aware diagnostics.

    Reporting convention:
    - headings use Markdown ATX headings;
    - structured, aligned data use ``_report_block()``;
    - prose uses ordinary Markdown/text;
    - environment-specific presentation is handled only by ``_render_report()``.
    """
    @staticmethod
    def _is_notebook() -> bool:
        """Return True when running in a Jupyter notebook or JupyterLab."""
        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell is None:
                return False
            shell_class = shell.__class__
            if shell_class.__name__ == "ZMQInteractiveShell":
                return True
            return shell_class.__module__.startswith("google.colab.")
        except (ImportError, NameError):
            return False

    def _report_block(self, text: str) -> str:
        """Wrap aligned report data in canonical Markdown text fencing."""
        return f"```text\n{text.rstrip()}\n```"

    def report(self, show: bool = True, **kwargs) -> str:
        """
        Returns a human-readable narrative report of the object's state.
        
        Args:
            show (bool): If True (default), displays the report appropriately 
                         for the environment (prints or renders Markdown).
            **kwargs: Configuration for the report generation.
        """
        raise NotImplementedError("Subclasses must implement .report()")

    def _render_report(self, content: str, show: bool = True, **kwargs) -> str:
        """
        Render canonical report Markdown according to the requested format and
        the active frontend.
        """
        markdown = kwargs.get("markdown")
        use_markdown = self._is_notebook() if markdown is None else bool(markdown)
        rendered = content if use_markdown else self._to_plain_sober(content)

        if not show:
            return rendered

        is_notebook = self._is_notebook()

        if is_notebook:
            try:
                from IPython.display import display, Markdown
                display(Markdown(rendered))
            except ImportError:
                print(self._decorate_report_content(self._to_plain_sober(rendered)))
                return rendered
            return _SilentDisplayedText(rendered)
        else:
            plain = self._to_plain_sober(rendered)
            print(self._decorate_report_content(plain))
            
        return rendered

    @staticmethod
    def _extract_report_title(content: str) -> str:
        """Extract a readable title from the first non-empty line."""
        for line in content.splitlines():
            title = line.strip()
            if not title:
                continue
            heading = re.match(r"^#{1,6}(?:\s+|$)", title)
            if heading:
                title = title[heading.end():].strip()
            if title.startswith("--- ") and title.endswith(" ---") and len(title) > 8:
                title = title[4:-4].strip()
            title = title.replace("**", "").replace("`", "")
            return title or "Report"
        return "Report"

    def _decorate_report_content(self, content: str) -> str:
        """Add visual separation for consecutive report calls."""
        title = self._extract_report_title(content)
        banner = "=" * max(len(title) + 8, 34)
        return f"{banner}\n{content}\n"

    @staticmethod
    def _to_plain_sober(content: str) -> str:
        """Convert MoeaBench's small canonical Markdown subset to plain text."""
        has_trailing_newline = content.endswith("\n")
        lines = []
        in_fence = False
        for line in content.splitlines():
            if not in_fence and re.match(r"^\s{0,3}```text\s*$", line):
                in_fence = True
                continue
            if in_fence:
                if re.match(r"^\s{0,3}```\s*$", line):
                    in_fence = False
                else:
                    lines.append(line)
                continue

            stripped = line.strip()
            heading = re.match(r"^\s{0,3}#{1,6}(?:\s+|$)", line)
            if heading:
                line = line[heading.end():]
            if stripped.startswith("--- ") and stripped.endswith(" ---") and len(stripped) > 8:
                line = stripped[4:-4].strip()
            line = line.replace("**", "")
            lines.append(line)
        text = "\n".join(lines)
        if has_trailing_newline:
            text += "\n"
        return text

    def report_show(self, **kwargs):
        """
        [DEPRECATED] Displays the report appropriately for the environment.
        Use .report(show=True) instead.
        """
        return self.report(show=True, **kwargs)

    def __repr__(self):
        # Concise representation that hints at report availability
        return f"<{self.__class__.__name__} (call .report() for narrative context)>"

    def _repr_pretty_(self, p, cycle):
        """Rich representation for Jupyter/IPython (Text)."""
        if cycle:
            p.text(str(self))
            return
        p.text(self.report(show=False, markdown=False))

    def _repr_markdown_(self):
        """Rich representation for Jupyter/IPython (Markdown)."""
        return self.report(show=False, markdown=True)


def emit_output(text: str, markdown: str | None = None) -> str:
    """
    Environment-aware output helper for non-report functions.
    Uses Markdown rendering in notebooks when provided.
    """
    is_notebook = Reportable._is_notebook()

    if is_notebook and markdown is not None:
        try:
            from IPython.display import display, Markdown
            display(Markdown(markdown))
            return text
        except ImportError:
            pass

    print(Reportable._to_plain_sober(text))
    return text
