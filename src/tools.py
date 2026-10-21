from langchain.tools import tool

from src.database import Database
from src.models import FinalAnswer


def build_tools(database: Database):
    @tool
    def list_tables() -> list[str]:
        """Lista as tabelas de negócio disponíveis, sem carregar seus dados."""
        return database.list_tables()

    @tool
    def inspect_schema(tables: list[str]) -> dict:
        """Descobre colunas, tipos, chaves e relações candidatas das tabelas selecionadas."""
        return database.inspect_schema(tables)

    @tool
    def execute_query(sql: str) -> dict:
        """Solicita um SELECT SQLite; o grafo valida e executa com limites e somente leitura."""
        return database.execute(sql, "direct").model_dump()

    @tool
    def finish(answer: FinalAnswer) -> dict:
        """Conclui com evidências identificadas, interpretação explícita e visualização."""
        return answer.model_dump()

    @tool
    def clarify(question: str) -> str:
        """Solicita ao usuário esclarecimento indispensável sobre período ou significado."""
        return question

    return [list_tables, inspect_schema, execute_query, finish, clarify]
