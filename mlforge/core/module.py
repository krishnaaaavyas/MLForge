from mlforge.core.parameter import Parameter


class Module:
    """
    Base class for all MLForge modules.

    Responsible for:

    - Tracking Parameters
    - Tracking child Modules
    - Returning parameters recursively
    """

    def __init__(self):

        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):

        if isinstance(value, Parameter):

            self.__dict__.setdefault("_parameters", {})
            self._parameters[name] = value

        elif isinstance(value, Module):

            self.__dict__.setdefault("_modules", {})
            self._modules[name] = value

        object.__setattr__(self, name, value)

    def parameters(self):

        params = []

        params.extend(self._parameters.values())

        for module in self._modules.values():
            params.extend(module.parameters())

        return params

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)