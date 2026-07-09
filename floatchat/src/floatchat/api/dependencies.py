"""FastAPI dependency injection.

All heavy dependencies are constructed lazily and cached so they can be
overridden easily in tests via ``app.dependency_overrides``.
"""

from typing import Annotated

from fastapi import Depends

from floatchat.conversation.base import AbstractConversationManager
from floatchat.conversation.memory import InMemoryConversationManager
from floatchat.intent_parser.base import AbstractIntentParser
from floatchat.intent_parser.regex import RegexIntentParser
from floatchat.query_normalizer.base import AbstractQueryNormalizer
from floatchat.query_normalizer.ollama import OllamaQueryNormalizer
from floatchat.llm_service.base import AbstractLLMService
from floatchat.llm_service.classifier import QueryClassifier
from floatchat.llm_service.ollama import OllamaLLMService
from floatchat.metadata_service.base import AbstractMetadataService
from floatchat.metadata_service.gdac import GDACMetadataService
from floatchat.netcdf_reader.base import AbstractNetCDFReader
from floatchat.netcdf_reader.bgc_reader import BGCNetCDFReader
from floatchat.query_engine.engine import QueryEngine
from floatchat.repository_service.base import AbstractRepositoryService
from floatchat.repository_service.gdac_http import GDACRepositoryService
from floatchat.visualization_engine.base import AbstractVisualizationEngine
from floatchat.visualization_engine.profile import ProfileVisualizationEngine

# Singleton caches (module-level for simplicity in MVP).
_metadata_service: GDACMetadataService | None = None
_repository_service: GDACRepositoryService | None = None
_netcdf_reader: BGCNetCDFReader | None = None
_viz_engine: ProfileVisualizationEngine | None = None
_intent_parser: RegexIntentParser | None = None
_query_engine: QueryEngine | None = None
_llm_service: OllamaLLMService | None = None
_query_classifier: QueryClassifier | None = None
_conversation_manager: InMemoryConversationManager | None = None
_query_normalizer: OllamaQueryNormalizer | None = None


def get_metadata_service() -> AbstractMetadataService:
    global _metadata_service
    if _metadata_service is None:
        _metadata_service = GDACMetadataService()
    return _metadata_service


def get_repository_service() -> AbstractRepositoryService:
    global _repository_service
    if _repository_service is None:
        _repository_service = GDACRepositoryService()
    return _repository_service


def get_netcdf_reader() -> AbstractNetCDFReader:
    global _netcdf_reader
    if _netcdf_reader is None:
        _netcdf_reader = BGCNetCDFReader()
    return _netcdf_reader


def get_visualization_engine() -> AbstractVisualizationEngine:
    global _viz_engine
    if _viz_engine is None:
        _viz_engine = ProfileVisualizationEngine()
    return _viz_engine

def get_query_normalizer() -> AbstractQueryNormalizer:
    global _query_normalizer

    if _query_normalizer is None:
        _query_normalizer = OllamaQueryNormalizer()

    return _query_normalizer

def get_intent_parser(
    normalizer: Annotated[
        AbstractQueryNormalizer,
        Depends(get_query_normalizer),
    ],
) -> AbstractIntentParser:
    global _intent_parser
    if _intent_parser is None:
        _intent_parser = RegexIntentParser(
    normalizer=normalizer
)
    return _intent_parser


def get_query_engine(
    metadata: Annotated[AbstractMetadataService, Depends(get_metadata_service)],
    repository: Annotated[AbstractRepositoryService, Depends(get_repository_service)],
    reader: Annotated[AbstractNetCDFReader, Depends(get_netcdf_reader)],
    viz: Annotated[AbstractVisualizationEngine, Depends(get_visualization_engine)],
) -> QueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine(metadata, repository, reader, viz)
    return _query_engine


def get_llm_service() -> AbstractLLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = OllamaLLMService()
    return _llm_service



def get_query_classifier(
    llm: Annotated[AbstractLLMService, Depends(get_llm_service)],
) -> QueryClassifier:
    global _query_classifier
    if _query_classifier is None:
        _query_classifier = QueryClassifier(llm)
    return _query_classifier


def get_conversation_manager() -> AbstractConversationManager:
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = InMemoryConversationManager()
    return _conversation_manager
