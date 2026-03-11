# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

class Reportable:
    """
    Mixin for objects that support narrative reporting in MoeaBench.
    Provides a consistent interface for environment-aware diagnostics.
    """
    @staticmethod
    def _is_notebook() -> bool:
        """Returns True when running inside IPython/Jupyter."""
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except (ImportError, NameError):
            return False

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
        Internal helper to handle the display logic of reports.
        """
        if not show:
            return content

        is_notebook = self._is_notebook()

        if is_notebook:
            try:
                from IPython.display import display, Markdown
                display(Markdown(content))
            except ImportError:
                print(content)
        else:
            print(self._to_plain_sober(content))
            
        return content

    @staticmethod
    def _to_plain_sober(content: str) -> str:
        """Light cleanup for terminal output to keep reports sober and readable."""
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("### "):
                line = stripped[4:]
            if stripped.startswith("--- ") and stripped.endswith(" ---") and len(stripped) > 8:
                line = stripped[4:-4].strip()
            line = line.replace("**", "")
            lines.append(line)
        return "\n".join(lines)

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
    try:
        from IPython import get_ipython
        is_notebook = get_ipython() is not None
    except (ImportError, NameError):
        is_notebook = False

    if is_notebook and markdown is not None:
        try:
            from IPython.display import display, Markdown
            display(Markdown(markdown))
            return text
        except ImportError:
            pass

    print(Reportable._to_plain_sober(text))
    return text
