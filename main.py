from corpmind.config import settings
import chromadb

client = chromadb.PersistentClient(path=settings.VECTOR_STORE_PATH)
client.delete_collection(name=settings.VECTOR_STORE_COLLECTION)
print("Collection dropped.")