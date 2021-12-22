import os
from abc import ABC, abstractmethod


class IndexerModel(ABC):
    def __init__(self):
        print("initialising indexer model")
        self.model = None

    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def compile_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

#    @abstractmethod
#    def save_model(self, weights_path):
#        pass

#    @abstractmethod
#    def load_model(self, weights_path):
#        pass


class Test(IndexerModel):
    def __init__(self):
        super().__init__()
        self.test = "hello"

    def build_model(self, *args, **kwargs):
        return "WIP"

    def compile_model(self, *args, **kwargs):
        return "WIP"

    def fit_model(self, *args, **kwargs):
        return "WIP"

    def get_model(self):
        return "WIP"

    def get_params(self):
        return "WIP"

    def save_model(self, weights_path):
        return "WIP"

    # def load_model(self, weights_path):
    #    return "WIP"
"""

def connect_to_bucket(
    project=os.environ["project"], bucket=os.environ["bucket"], mode="Cloud-Dev"
):
    from google.cloud import storage

    try:
        print("Getting Cloud Credentials and setting up client.")
        client = None
        if mode == "Cloud-Dev":
            client = storage.Client(project=project)
        elif mode == "Cloud-Prod":
            from google.auth import compute_engine

            credentials = compute_engine.Credentials()
            client = storage.Client(credentials=credentials, project=project)
        else:
            print(
            #
            #Invalid Mode: valid mode param is one of these
            #:["Cloud-Dev", "Cloud-Prod"]
            #)
        bucket = client.get_bucket(bucket)
        print("Successfully connected to Cloud Storage api.")
        return bucket
    except Exception as e:
        print("Error Connecting to Cloud Storage. Error: {}".format(e))
        raise e

def load_model_interface(mode):
    print("Running Load Model Decorator")

    def inner(func):
        def wrapper(*args, **kwargs):
            import tempfile

            try:
                test_mode = mode.split("-")[0]
                if test_mode == "Cloud":
                    bucket = connect_to_bucket(mode=mode)
                    print(kwargs)
                    blob = bucket.get_blob(kwargs["weights_path"])
                    with tempfile.NamedTemporaryFile() as temp_file:
                        blob.download_to_file(temp_file)
                        temp_file.seek(0)
                        print(kwargs)
                        kwargs["weights_path"] = temp_file.name
                        func(*args, **kwargs)
                elif mode == "Local":
                    func(*args, **kwargs)
                else:
                    print("Define a valid mode param for the function")
                    raise ValueError(
                        #["Cloud-Prod", "Cloud-Dev", "Local"] are the options
                    )
            except Exception as e:
                raise e

        return wrapper

    return inner


def save_model_interface(mode):
    print("Running Save Model Decorator")

    def inner(func):
        def wrapper(*args, **kwargs):
            try:
                test_mode = mode.split("-")[0]
                if test_mode == "Cloud":
                    bucket = connect_to_bucket(mode=mode)
                    weights_path = kwargs["weights_path"]
                    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                    weights_blob = bucket.blob(weights_path)
                    print("Saving Model to Local")
                    func(*args, **kwargs)
                    print("Saving Model to Cloud")
                    weights_blob.upload_from_filename(weights_path)
                elif mode == "Local":
                    print("Saving Model to Local")
                    weights_path = kwargs["weights_path"]
                    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                    func(*args, **kwargs)
                else:
                    print("Define a valid mode param for the function")
                    raise ValueError(
                        #["Cloud-Prod", "Cloud-Dev", "Local"] are the options
                    )
            except Exception as e:
                raise e

        return wrapper

    return inner
"""


if __name__ == "__main__":
    test = Test()
