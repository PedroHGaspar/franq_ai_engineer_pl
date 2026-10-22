# Franq · Assistente Virtual de Dados

Implementação do Desafio 1: investigação de um SQLite por perguntas em português,
com SQL dinâmico, evidências, tabelas e gráficos. O banco privado não é distribuído.

## Tecnologias

Python 3.12 permite tipagem e testes simples; LangGraph controla o ciclo de investigação;
LangChain integra o modelo e as ferramentas; SQLite dispensa serviços externos;
Streamlit oferece a interface local. Pydantic valida contratos, SQLGlot inspeciona SQL,
Pandas/Plotly apresentam resultados e Pytest/Ruff verificam o projeto.

## Configuração e execução

Na raiz do projeto, com Python 3.12 instalado:

```bash
python -m venv .venv
```

Ative com `.venv\Scripts\Activate.ps1` no PowerShell ou `source .venv/bin/activate`
no Linux/WSL. Depois:

```bash
python -m pip install -r requirements.lock
python -c "from pathlib import Path; p=Path('.env'); p.exists() or p.write_bytes(Path('.env.example').read_bytes())"
python -m streamlit run app.py --server.address 127.0.0.1
```

Abra **http://localhost:8501** no navegador do Windows, inclusive ao executar no WSL.
Mantenha o terminal aberto; `Ctrl+C` encerra o servidor. A aplicação escuta apenas localmente.
Após atualizar o código (ex.: `git pull`) ou o `.env`, encerre com `Ctrl+C` e execute de novo:
o Streamlit mantém em memória os módulos de `src/` já carregados.
Períodos relativos usam os dados mais recentes do banco: “em maio” sem ano é o maio mais
recente com registros e “último ano” são os 12 meses até o último registro. A resposta informa
o período considerado.
No primeiro início, deixe o campo de e-mail do Streamlit vazio, se solicitado.

Coloque `anexo_desafio_1.db` na raiz ou ajuste `DATABASE_PATH` no `.env`.
Configure `OPENAI_API_KEY` apenas localmente. `ENABLE_PAID_API=false` é o padrão:
a interface abre, mas não envia perguntas. Habilite somente após autorizar o uso pago.
Não é necessário configurar chave para executar testes.

Modelo inicial: `gpt-5.4-mini-2026-03-17`, configurável por `OPENAI_MODEL`.
O teto local inicial é US$ 0,50, independente do orçamento mensal de US$ 5.
Uma reserva conservadora precede cada chamada; consumo conhecido substitui a reserva,
e falhas sem informação de uso mantêm a reserva. O controle persiste em `.local/usage.sqlite`.
Não apague esse arquivo durante a validação. Ao mudar o modelo, atualize também
`INPUT_PRICE_PER_MILLION` e `OUTPUT_PRICE_PER_MILLION`; o controle depende dessas tarifas.

## Arquitetura

```mermaid
flowchart TD
    A[Pergunta] --> B[Decidir ação]
    B -->|Ferramentas de schema| C[Metadados]
    C --> F[Avaliar]
    B -->|SQL| D[Validar]
    D -->|Autorizado| E[Executar]
    D -->|Erro| F
    E --> F
    F -->|Continuar ou corrigir| B
    F -->|Bloqueio ou limite| X[Falha auditável]
    B -->|Ambiguidade| G[Esclarecer]
    G -. Resposta do usuário .-> B
    B --> H[Validar resposta e evidências]
    H -->|Inconsistência| F
    H -->|Válido| I[Texto, tabela e gráfico]
```

`src/agent.py` implementa o `StateGraph`; `tools.py` expõe ferramentas LangChain;
`database.py` descobre tabelas, colunas, FKs e relações candidatas sem schema fixo.
`models.py` define os contratos; `config.py` centraliza limites;
`visualization.py` valida gráficos; `observability.py` registra eventos e consumo.
Checkpoints em memória retomam esclarecimentos na sessão, sem persistir conversas em disco.

SQL passa por SQLGlot, conexão `mode=ro`, `query_only` e autorizador SQLite.
Somente SELECT e funções autorizadas; escrita, múltiplas instruções, extensões,
ATTACH, PRAGMA do modelo e CTE recursiva são bloqueados. Há limites de tempo,
linhas, bytes, consultas, correções, contexto e chamadas. Erros recuperáveis voltam
ao modelo; violações de segurança encerram a execução. Vazio, truncamento e
inconsistência têm tratamentos distintos.

Tabelas e gráficos usam somente resultados executados. A resposta exige IDs válidos
e verifica números contra dados/filtros, aceitando arredondamento decimal validado
deterministicamente; isso não prova a semântica da análise.
O expansor mostra ferramentas, SQL, erros, tentativas e duração, sem raciocínio privado.
Logs JSON locais omitem prompts, SQL e linhas individuais. Histórico e consultas ficam
na sessão; não use a interface como serviço público sem controles adicionais.
Gráficos identificam sua consulta de origem, com eixos e séries validados; dados incompatíveis
permanecem em tabela. Esclarecimentos são respondidos no mesmo campo de perguntas.

Antes do modelo, meses sem ano e “último ano/semestre/trimestre/mês” são resolvidos pela
cobertura de datas descoberta no banco, e o intervalo é informado ao modelo. Ao concluir,
a evidência principal precisa ser compatível com a intenção: contagens e comparações exigem agregação, tendências exigem agrupamento por
período (e pela dimensão pedida), rankings exigem ordenação, anos e meses pedidos devem constar
nos filtros e contagens de entidades fora de sua tabela exigem `COUNT(DISTINCT)`. Consultas de
descoberta não são aceitas como resposta; sem evidência suficiente dentro dos limites, a
interface informa que a análise não foi concluída. Cada pergunta tem ID e estado próprios;
perguntas seguintes recebem apenas perguntas e sínteses anteriores, nunca SQL ou linhas.

## Perguntas e testes

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m pip check
```

Os testes bloqueiam conexões de rede e usam modelos/HTTP simulados. Cobrem segurança
adversarial, descoberta, correção, múltiplas consultas, limites, evidências, gráficos,
orçamento e interface Streamlit. Testes `original` usam o banco privado quando presente;
sem ele, são explicitamente ignorados. As fixtures pequenas são exclusivas dos testes.

Referências verificadas por SQL local (não por LLM real):

- Clientes que interagiram com WhatsApp em 2024: **17**.
- Reclamações não resolvidas: **Chat 18, E-mail 14, Telefone 19**.
- Estados com mais clientes via App em **maio de 2025**, com desempate alfabético.
- Média de compras por **comprador de cada categoria**, comparada a cálculo Python.
- Reclamações mensais por canal entre **23/07/2024 e 22/07/2025**, comparadas a Python.
- Pergunta inédita: clientes que nunca compraram pelo App, validada por diferença de conjuntos.

“Maio” e “último ano” são resolvidos pelos dados disponíveis; o denominador de médias é
declarado na interpretação adotada.
Os resumos de gasto/última compra em `clientes` divergem dos eventos em `compras`;
as análises de compras devem considerar essa limitação. O agente deve descobrir
essas condições pelos dados, não por consultas predefinidas no código de produção.

## Limitações e melhorias

Uma primeira bateria real conferiu WhatsApp, reclamações e médias por categoria com SQL
independente. A resposta de médias foi revalidada offline após corrigir um falso positivo
de arredondamento. A API exigiu `reasoning_effort="none"` para ferramentas nesse modelo
via Chat Completions. Avaliação ampliada de perguntas inéditas permanece pendente.
A interface foi inspecionada em Chromium desktop/mobile com simulações e respostas salvas.
A bateria complementar comprovou seleção real de barras e linhas, com SQL e pontos conferidos.
O texto da tendência apresentou uma afirmação não sustentada sobre canais ausentes em julho/2025;
essa resposta textual não foi aprovada, embora consulta e gráfico estejam corretos.
Agora a apresentação usa sínteses determinísticas dos resultados (métrica, ranking, comparação
ou primeiro/último/pico de cada série) e mantém o texto original do modelo na transparência,
com números conferidos. As regras de intenção são heurísticas lexicais em português e podem
não reconhecer formulações incomuns. A bateria conversacional usa modelos simulados; as
correções ainda não foram revalidadas com o LLM real. O banner local é opcional; veja
`assets/README.md`.
As checagens numéricas não detectam toda
conclusão semântica incorreta; comparar resposta, SQL e referência continua necessário.
As relações inferidas por nome/tipo são hipóteses. Checkpoints desaparecem ao encerrar
o processo. Tarifas são configuráveis e o orçamento local não cobre uso externo à aplicação.

Próximas melhorias: avaliação real progressiva, catálogo de métricas aprovado pelo negócio,
checagens semânticas adicionais e política de retenção. Dependências estão fixadas em
`uv.lock`; `requirements.lock` é sua exportação para instalação com pip.
