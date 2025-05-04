import pytest
from src.retrieval import AIDocumentStore

# Test that the constructor correctly sets initial attributes
def test_aidocumentstore_initialization():
    store = AIDocumentStore(dataset_path="dummy.csv", index_path="dummy.index", chunk_size=100)
    assert store.dataset_path == "dummy.csv"
    assert store.index_path == "dummy.index"
    assert store.chunk_size == 100
    assert isinstance(store.documents, list)
    assert isinstance(store.document_metadata, list)

# Test that load_and_split() raises an error if file doesn't exist
def test_load_and_split_file_not_found():
    with pytest.raises(FileNotFoundError) as excinfo:
        store = AIDocumentStore(dataset_path="nonexistent_file.csv", index_path="dummy.index")
        store.load_and_split()
    assert "Dataset not found" in str(excinfo.value)

# Test that chunk_text() properly splits a large string into expected chunks
def test_chunk_text_basic_split():
    text = "word " * 1050
    store = AIDocumentStore(dataset_path=None, index_path=None)
    chunks = store.chunk_text(text, size=500)
    assert len(chunks) == 3
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert sum(len(chunk.split()) for chunk in chunks) == 1050

# Test chunk_text() handles shorter input correctly
def test_chunk_text_small_text():
    text = "tiny text"
    store = AIDocumentStore(dataset_path=None, index_path=None)
    chunks = store.chunk_text(text, size=500)
    assert len(chunks) == 1
    assert chunks[0] == text

# Test chunk_text() returns an empty list on empty input
def test_chunk_text_empty_string():
    text = ""
    store = AIDocumentStore(dataset_path=None, index_path=None)
    chunks = store.chunk_text(text, size=500)
    assert chunks == []

# Test chunk_text() creates expected word groupings
def test_chunk_text_words_are_correct():
    text = "one two three four five six seven eight nine ten"
    store = AIDocumentStore(dataset_path=None, index_path=None)
    chunks = store.chunk_text(text, size=3)
    assert chunks == ["one two three", "four five six", "seven eight nine", "ten"]

# Test get_documents() returns a list of document chunks
def test_get_documents_returns_list():
    store = AIDocumentStore(dataset_path=None, index_path=None)
    store.documents = ["doc1", "doc2", "doc3"]
    docs = store.get_documents()
    assert isinstance(docs, list)
    assert docs == ["doc1", "doc2", "doc3"]

# Test get_metadata() returns a list of metadata entries
def test_get_metadata_returns_list():
    store = AIDocumentStore(dataset_path=None, index_path=None)
    store.document_metadata = [{"title": "Paper 1", "url": "http://example.com/1"}]
    metadata = store.get_metadata()
    assert isinstance(metadata, list)
    assert metadata[0]["title"] == "Paper 1"

# Test that load_index() raises an error if the FAISS index is missing
def test_load_index_raises_error_if_not_found():
    store = AIDocumentStore("dummy.csv", "nonexistent.index")
    with pytest.raises(FileNotFoundError):
        store.load_index()