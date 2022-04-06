from typing import Callable, Iterator, List, Optional
from mongoengine import Document
from pymongo.collection import Collection


def get_random_oids(collection: Collection, sample_size: int) -> List[int]:
    pipeline = [{"$project": {"_id": 1}}, {"$sample": {"size": sample_size}}]
    return [s["_id"] for s in collection.aggregate(pipeline)]


def get_random_documents(DocCls: Document, sample_size: int) -> Iterator[Document]:
    doc_collection = DocCls._get_collection()
    random_oids = get_random_oids(doc_collection, sample_size)
    return DocCls.objects(id__in=random_oids)


def check_documents(DocCls: Document, sample_size: int, custom_validation: Optional[Callable[[Document], bool]]):
    for doc in get_random_documents(DocCls, sample_size):
        # general validation (types and values)
        doc.validate()

        # load all subfields,
        # this may trigger additional queries if you have ReferenceFields
        # so it may be slow
        for field in doc._fields:
            try:
                getattr(doc, field)
            except Exception:
                print(f"Could not load field {field} in Document {doc.id}")
                raise

        # custom validation
        if custom_validation is not None:
            custom_validation(doc)
