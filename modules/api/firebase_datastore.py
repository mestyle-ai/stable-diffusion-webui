import firebase_admin
import uuid
from firebase_admin import firestore
from firebase_admin import credentials

CRED_JSON_FILE = "/home/ubuntu/.firebase/mestyle-cred.json"
# CRED_JSON_FILE = "/Users/apirat/.firebase/mestyle-cred.json"

class DataStore:

    db = None

    """
    Initializze firebase object
    """
    def __init__(self):
        try:
            app = firebase_admin.get_app()
        except ValueError as e:
            cred = credentials.Certificate(CRED_JSON_FILE)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    """
    Retrieve document by document's key
    """
    def get_doc(self, collection: str, key: str):
        ref = self.db.collection(collection)
        doc = ref.document(key).get()
        
        return doc.to_dict()

    """
    Store document into Firebase datastore
    """
    def set_doc(self, collection: str, key: str, data: dict):
        ref = self.db.collection(collection)
        doc = ref.document(key)
        doc.set(data)

        return True


if __name__ == "__main__":
    pass
    # ds = DataStore()
    # doc = ds.get_doc(collection="models", key="134c8678-fdd6-4109-878d-5d44140c8bc3")
    # print(doc)
    # doc["modelDescription"] = doc["modelDescription"] + "."
    # ds.set_doc(
    #     collection="models",
    #     key="134c8678-fdd6-4109-878d-5d44140c8bc3",
    #     data=doc,
    # )
