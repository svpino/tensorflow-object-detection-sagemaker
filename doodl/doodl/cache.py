import os
import numpy as np

from doodl import Configuration


class Cache:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.enabled = configuration.cache is True
        self.cache_path = configuration.cache_path or "."

    def get(self, key: str):
        if self.enabled:
            entry = os.path.join(self.cache_path, f"{key}.npz")
            if os.path.exists(entry):
                data = np.load(entry, allow_pickle=True)
                return data["data"]

        return None

    def put(self, key: str, value):
        if self.enabled:
            np.savez_compressed(os.path.join(self.cache_path, f"{key}.npz"), data=value)
