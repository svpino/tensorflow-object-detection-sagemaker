class Configuration:
    def __init__(self, data: dict = None):
        self.endpoint = data.get("endpoint", None) if data else None
        self.backend = data.get("backend", None) if data else None

        self.model = data.get("model", None) if data else None
        self.model_path = data.get("model_path", None) if data else None
        self.model_label_path = data.get("model_label_path", None) if data else None

        self.cache = data.get("cache", None) if data else None
        self.cache_path = data.get("cache_path", None) if data else None
        self.cache_id = data.get("cache_id", None) if data else None

        self.aws_region = data.get("aws_region", None) if data else None
        self.aws_access_key = data.get("aws_access_key", None) if data else None
        self.aws_secret_access_key = (
            data.get("aws_secret_access_key", None) if data else None
        )
