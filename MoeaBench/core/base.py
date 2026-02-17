# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

class Reportable:
    """
    Mixin for objects that support narrative reporting in MoeaBench.
    Provides a consistent interface for environment-aware diagnostics.
    """
    def report(self, **kwargs) -> str:
        """Returns a human-readable narrative report of the object's state."""
        raise NotImplementedError("Subclasses must implement .report()")

    def report_show(self, **kwargs):
        """
        Displays the report appropriately for the environment.
        Prints clean text to console in scripts, renders Markdown in Notebooks.
        """
        # Check if running in Jupyter/IPython
        try:
            from IPython import get_ipython
            is_notebook = get_ipython() is not None
        except (ImportError, NameError):
            is_notebook = False

        # Request appropriate format from the report() implementation
        # (Defaulting to markdown=True for backward compatibility if not explicitly handled)
        content = self.report(markdown=is_notebook, **kwargs)
        
        if is_notebook:
            try:
                from IPython.display import display, Markdown
                display(Markdown(content))
            except ImportError:
                print(content)
        else:
            print(content)

    def __repr__(self):
        # Concise representation that hints at report availability
        return f"<{self.__class__.__name__} (call .report() or .report_show() for context)>"

    def _repr_pretty_(self, p, cycle):
        """Rich representation for Jupyter/IPython."""
        if cycle:
            p.text(str(self))
            return
        p.text(self.report())
