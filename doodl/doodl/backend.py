from doodl import Configuration


class Backend:
    backends = {}

    @staticmethod
    def register(backend: str, configuration: Configuration):
        if not backend or backend.lower() == "tensorflow":
            backend_key = f"tensorflow-{configuration.model_path}-{configuration.model}"
            if backend_key not in Backend.backends:
                try:
                    from doodl_tensorflow.backend import TensorflowBackend

                    backend = TensorflowBackend(
                        model=configuration.model,
                        model_path=configuration.model_path,
                        model_label_path=configuration.model_label_path,
                    )

                    Backend.backends[backend_key] = backend

                except Exception:
                    raise RuntimeError(
                        "Unable to find tensorflow backend implementation. "
                        "Make sure you have 'doodl[tensorflow]' installed"
                    )

            return Backend.backends[backend_key]

        else:
            backend_key = f"{backend}"
            if backend_key not in Backend.backends:
                try:
                    Backend.backends[backend_key] = Backend._class_from_str(backend)(
                        configuration
                    )
                except Exception:
                    raise RuntimeError(
                        f"There was en error creating backend " f"{backend}"
                    )

            return Backend.backends[backend_key]

    @staticmethod
    def _class_from_str(classname):
        from importlib import import_module

        module_path, _, class_name = classname.rpartition(".")
        module = import_module(module_path)
        return getattr(module, class_name)

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def inference(self, image):
        pass
