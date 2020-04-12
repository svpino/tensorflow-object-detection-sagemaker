import os
import numpy as np

from doodl import Configuration


class Cache:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.enabled = configuration.cache is True

        if "DOODL_HOME" in os.environ:
            doodl_dir = os.environ.get("DOODL_HOME")
        else:
            doodl_dir = os.path.join(os.path.expanduser("~"), ".doodl")

        cache_base_dir = os.path.expanduser(doodl_dir)

        if not os.access(cache_base_dir, os.W_OK):
            cache_base_dir = os.path.join("/tmp", ".doodl")

        self.cache_path = os.path.join(cache_base_dir, "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

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
