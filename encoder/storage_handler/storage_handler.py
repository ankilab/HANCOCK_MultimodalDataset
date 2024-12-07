from pathlib import Path
import chromadb
import numpy as np
import warnings
import sys
sys.path.append(str(Path(__file__).parents[2]))
from defaults import DefaultPaths


class DataHandler:
    def __init__(self):
        pass

    def save_file(
            self, base_path: Path, file_id: str, file_name: str,
            file_extension: str, file: any, create_path=False
    ) -> bool:
        raise NotImplementedError('This method must be implemented by the subclass')

    def load_file(
            self, base_path: Path, file_id: str, file_name: str,
            file_extension: str
    ) -> any:
        raise NotImplementedError('This method must be implemented by the subclass')

    @staticmethod
    def _create_path(base_path: Path, file_id: str, file_name: str,
                     file_extension: str) -> Path:
        return base_path / file_id / f'{file_name}{file_extension}'


class NumpyTensorHandler(DataHandler):
    def save_file(
            self, base_path: Path, file_id: str, file_name: str,
            file_extension: str, file: any, create_path=False
    ) -> bool:
        save_path = self._create_path(base_path, file_id, file_name, file_extension)
        if save_path.suffix != '.npy':
            warnings.warn(f'File {save_path} does not have a .npy extension')
            return False

        if create_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            if not save_path.parent.exists():
                warnings.warn(f'Parent directory {save_path.parent} does not exist')
                return False

        np.save(save_path, file)
        return True

    def load_file(
            self, base_path: Path, file_id: str, file_name: str,
            file_extension: str
    ) -> np.ndarray | None:
        load_path = self._create_path(base_path, file_id, file_name, file_extension)
        path_suffix = load_path.suffix
        if path_suffix and path_suffix == '.npy':
            if load_path.exists():
                data = np.load(load_path)
                return data
        else:
            warnings.warn(f'The file {load_path} does not have a file suffix')
        return None


class ChromaTensorHandler(DataHandler):
    def __init__(self, chroma_db_path: Path = DefaultPaths().encodings_db_path):
        super().__init__()
        self.chroma_db_path = chroma_db_path
        self.client = chromadb.PersistentClient(path=str(chroma_db_path))

    def save_file(
            self, base_path: Path, file_id: str, file_name: str,
            file_extension: str, file: any, create_path=False
    ) -> bool:
        if base_path != self.chroma_db_path:
            if create_path:
                self.chroma_db_path = base_path
                if not base_path.exists:
                    base_path.mkdir()
                self.client = chromadb.PersistentClient(path=str(base_path))
            else:
                warnings.warn(f'ChromaDB path {self.chroma_db_path} does not match the base path {base_path}')
                return False
        if file_extension != ".db":
            warnings.warn(f'File extension {file_extension} is not supported for ChromaDB')
            return False
        collection = self.client.get_or_create_collection(name=file_name)
        collection.add(ids=[file_id], embeddings=[file])
        return True

    def load_file(self, base_path: Path, file_id: str, file_name: str,
                  file_extension: str) -> np.ndarray | None:
        if not base_path.exists():
            warnings.warn(f'Base path {base_path} does not exist')
            return None

        if file_extension != ".db":
            warnings.warn(f'File extension {file_extension} is not supported for ChromaDB')
            return None
        try:
            collection = self.client.get_collection(name=file_name)
        except Exception as ex:
            warnings.warn(f'Error loading collection {file_name}: {ex}')
            return None
        result = collection.get(ids=[file_id], include=['embeddings'])['embeddings']

        if result is not None and len(result) == 1:
            return result[0]

        return None



