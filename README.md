# 🚀 Pipeline de Engenharia de Dados & Feature Store com Apache Hamilton

Este projeto implementa a estrutura completa e a arquitetura de uma solução corporativa de Engenharia de Dados e Feature Engineering (com foco em detecção de Churn) utilizando o **Apache Hamilton**.

O projeto demonstra a separação limpa entre a definição das transformações de dados (funções Python puras fortemente tipadas) e o driver de execução (orquestrador), garantindo rastreabilidade automática de linhagem (*lineage*), facilidade de testes unitários offline, validação de schemas em tempo de execução e integração nativa com a Hamilton UI e Jupyter Notebooks.

---

## 🏗️ 1. Arquitetura do Sistema & Paradigma do Hamilton

### Diagrama ASCII da Arquitetura
Abaixo, ilustramos como o Apache Hamilton separa as definições funcionais, o Driver de Execução e o grafo de execução (DAG) gerado dinamicamente:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MÓDULOS DE FUNÇÕES PURAS                               │
│  Define o QUE as transformações fazem. Sem acoplamento com IO ou motores de execução. │
│                                                                                        │
│     features/data_prep.py                features/feature_engineering.py               │
│   ┌───────────────────────────┐        ┌────────────────────────────────────────┐      │
│   │ raw_data() -> df          │        │ is_senior(imputed_age) -> series       │      │
│   │ imputed_age(age) -> series│        │ avg_monthly_spend(total, tenure) -> S  │      │
│   └───────────────────────────┘        └────────────────────────────────────────┘      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ (Passa os módulos)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              DRIVER DE EXECUÇÃO (run.py)                               │
│  Orquestra as execuções, injeta variáveis de ambiente, configs e seleciona os sinks. │
│                                                                                        │
│     Instancia o Driver:                                                                │
│     dr = driver.Builder().with_config(cfg).with_modules(prep, fe).build()              │
│                                                                                        │
│     Executa nós específicos:                                                           │
│     dr.execute(["final_feature_matrix"])                                               │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ (Compila em tempo de execução)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                DAG DE EXECUÇÃO DINÂMICA                                │
│      O Hamilton conecta os parâmetros de entrada aos nomes de funções equivalentes    │
│                                                                                        │
│                    [raw_data] ──▶ [age] ──▶ [mean_age]                                 │
│                         │                      │                                       │
│                         ▼                      ▼                                       │
│                    [total_charges]        [imputed_age] ──▶ [is_senior]                │
│                         │                      │                 │                     │
│                         ▼                      ▼                 ▼                     │
│               [imputed_total_charges] ────────────────────▶ [final_feature_matrix]     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ├──────────────────────────┐
                                            ▼                          ▼
                         ┌──────────────────────┐         ┌──────────────────────┐
                         │   TELEMETRIA & UI    │         │     SINKS / SAÍDA    │
                         │ Hamilton UI (8242)   │         │ Parquet / SQLite     │
                         │ HamiltonTracker      │         │ Delta Lake           │
                         └──────────────────────┘         └──────────────────────┘
```

### Paradigma "Function Name = Output Node Name"
No Hamilton, a fiação (wiring) de dependências é resolvida em tempo de execução inspecionando a assinatura de tipo das funções:
* O **nome de uma função** define o nome do nó de saída que ela gera na DAG.
* Os **parâmetros de entrada** de uma função definem suas dependências upstream.
* Exemplo: Se a função `is_senior(imputed_age: pd.Series)` depende de `imputed_age`, o Hamilton procurará um nó ou função chamado `imputed_age` na DAG para injetar seu resultado automaticamente.

### Decoradores Avançados Utilizados
1. `@config.when(market_condition="...")`: Permite criar ramificações lógicas condicionais no grafo com base em configurações passadas no instanciamento do Driver (ex: alterando a lógica de precificação sob inflação alta).
2. `@extract_columns("col1", "col2", ...)`: Quando uma função retorna um DataFrame, este decorador divide-o em múltiplas séries independentes de forma transparente na DAG.
3. `@check_output_custom(...)`: Garante qualidade de dados rígida (*Data Quality Gates*) em tempo de execução, lançando exceções se a saída violar o contrato de dados tipado especificado com um validador de schema personalizado extremamente robusto.
4. `@tag(feature_category="...")`: Classifica metadados dos nós para auditoria e governança no catálogo do Hamilton UI.

---

## 🛠️ 2. Infraestrutura & Containerização (`docker-compose.yml`)

Nossa arquitetura de serviços em container inclui tudo o que é necessário para desenvolver, rodar e analisar o pipeline em um ambiente autônomo de MLOps:

```yaml
# hamilton-data-pipeline/docker-compose.yml
# Inclui:
# - db: PostgreSQL 15 para persistência de metadados de telemetria
# - backend: Servidor Django do Hamilton UI coletando logs
# - frontend: Visualizador do Hamilton UI (DAG, linhagem, catálogo) rodando na porta 8242
# - pipeline-sandbox: Ambiente Python com JupyterLab para exploração e execução de DAGs
```

### Para Iniciar o Ambiente:
A partir da pasta `hamilton-data-pipeline/`, execute:
```bash
docker compose up -d
```
Acesse as ferramentas:
* **Hamilton UI:** `http://localhost:8242` (digite qualquer email para autenticação inicial e crie o Projeto ID `1`).
* **Jupyter Notebook Sandbox:** `http://localhost:8888` (para prototipação e exploração).

---

## 📦 3. Estrutura de Arquivos do Projeto

A solução está organizada de forma altamente modular seguindo as melhores práticas do ecossistema Python de MLOps:

```
hamilton-data-pipeline/
├── features/
│   ├── __init__.py
│   ├── data_prep.py            # Módulo com transformações puras (Limpeza, nulos, normalização)
│   └── feature_engineering.py  # Módulo com transformações avançadas, decoradores e tags
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # Suíte de testes unitários offline e testes de integração de DAG
├── output/                     # Diretório de artefatos gerados (Parquet, Gráficos PNG)
├── docker-compose.yml          # Definição do ecossistema de serviços (Postgres + UI + Sandbox)
├── requirements.txt            # Dependências de bibliotecas de dados e machine learning
├── run.py                      # Driver orquestrador (Hamilton Driver)
└── README.md                   # Guia Operacional e Documentação Arquitetural
```

---

## 🚀 4. Guia Operacional & Como Executar o Pipeline

### Passo 1: Configurar Ambiente Virtual Local (Opcional)
Se preferir rodar localmente fora do Docker Compose, configure um ambiente virtual:
```bash
cd hamilton-data-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Passo 2: Executar o Pipeline via Driver
O script `run.py` expõe parâmetros de CLI para configurar as regras do pipeline dinamicamente:

1. **Execução Padrão (Stable Market):**
   ```bash
   python run.py
   ```
2. **Execução para Cenário de Alta Inflação (Ativando `@config.when`):**
   ```bash
   python run.py --market_condition high_inflation
   ```
3. **Executar gerando o Diagrama Visual da DAG (.png):**
   ```bash
   python run.py --visualize
   ```
   *Nota: O grafo renderizado será salvo em `output/hamilton_dag.png`.*

4. **Executar enviando métricas reais para a Hamilton UI (Orquestração Ativa):**
   ```bash
   # Certifique-se de que o Docker Compose está rodando e você instalou a hamilton-sdk
   python run.py --enable_tracker
   ```

---

## 🧪 5. Validação de Dados, Testes e Resiliência

### 1. Garantia de Qualidade de Dados (Data Quality Gates)
O nó final `final_feature_matrix` no arquivo `features/feature_engineering.py` utiliza o decorador `@check_output_custom` associado ao nosso `DataFrameSchemaValidator` para impor verificação estrita de schema de dados do Pandas DataFrame em tempo de execução:
```python
@check_output_custom(DataFrameSchemaValidator(EXPECTED_SCHEMA, importance="fail"))
```
Se qualquer função upstream introduzir uma tipagem incorreta ou valores nulos (NaN) inesperados que fujam desse schema, o Hamilton abortará a execução imediatamente para evitar que dados corrompidos alimentem o modelo de Machine Learning downstream.

### 2. Rodando Testes Unitários de Funções Puras com `pytest`
Por serem funções funcionais puras sem acoplamento a instâncias de grafos ou bancos de dados, o teste unitário torna-se incrivelmente simples. Não há necessidade de mocks complexos:
```bash
pytest tests/test_pipeline.py -v
```

---

## 💡 6. Boas Práticas & Escalabilidade (Polars / PySpark)

### Organização e Versionamento de DAGs no Hamilton
* **Definições Puras vs. Infraestrutura:** Mantenha os arquivos sob a pasta `features/` totalmente livres de carregamento de arquivos externos, conexões SQL ou caminhos estáticos em disco. O carregamento de dados brutos deve ocorrer de forma abstrata por parâmetros injetados no Driver ou por adaptadores de entrada.
* **Granularidade Inteligente:** Evite criar funções monolíticas gigantescas. Cada operação (imputação de nulos, normalização, log de escala) deve ser uma função distinta para maximizar o reuso de código e permitir que o Hamilton entenda a linhagem exata da feature.

### Alternando Engines Dinamicamente (Pandas ➡️ Polars ➡️ PySpark)
Um dos maiores trunfos do Apache Hamilton é a capacidade de reescrever ou otimizar o processamento para grandes volumes de dados de maneira transparente.

Para alternar o backend de execução de **Pandas** para **Polars** ou **PySpark**, não é necessário modificar sequer uma linha de código das transformações declaradas em `features/`! Basta modificar o Driver de execução em `run.py`:

```python
# 1. Adaptador Padrão (Pandas)
from hamilton import driver
dr = driver.Builder().with_modules(data_prep, feature_engineering).build()

# 2. Utilizando Polars (Processamento Vetorial In-Memory Ultra Rápido)
# Basta carregar os módulos definidos para polars e configurar o ResultBuilder correspondente
from hamilton.plugins import polars_extensions
dr = (
    driver.Builder()
    .with_modules(data_prep, feature_engineering)
    .with_result_builder(polars_extensions.PolarsDataFrameResultBuilder())
    .build()
)

# 3. Escalando para PySpark UDFs (Processamento Distribuído em Clusters)
from hamilton.plugins import h_spark
dr = (
    driver.Builder()
    .with_modules(data_prep, feature_engineering)
    .with_adapters(h_spark.PySparkUDFGraphAdapter()) # Converte as funções automaticamente em UDFsSpark
    .build()
)
```

Isso garante que sua equipe possa testar pipelines localmente com pequenas amostras usando Pandas ou Polars, e promover exatamente as mesmas lógicas funcionais para produção rodando sobre clusters robustos de Databricks ou Apache Spark sem retrabalho!
