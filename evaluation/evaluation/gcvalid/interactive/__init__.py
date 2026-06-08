
import ipywidgets as widgets


class NocatchOutput(widgets.Output):
    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)
