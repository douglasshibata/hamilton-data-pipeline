"""
Driver de Execução do Apache Hamilton.

Este script atua como o ponto de entrada principal (Orchestrator) do pipeline:
1. Carrega os módulos Python declarativos definindo as transformações.
2. Inicializa o `driver.Driver` com as configurações do mercado.
3. Se especificado (por meio do Hamilton Tracker), registra telemetria e execuções no Hamilton UI.
4. Executa a DAG obtendo a matriz final de features desejada.
5. Gera visualizações em imagem (.png) representando o fluxo de linhagem e o grafo.
"""

import os
import argparse
import logging
from typing import Dict, Any

from hamilton import driver
from features import data_prep, feature_engineering

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(market_condition: str, visualize: bool = False, enable_tracker: bool = False) -> None:
    """Instancia o driver do Hamilton, executa a DAG e opcionalmente visualiza o grafo."""

    # 1. Configurações iniciais injetadas na DAG
    config = {
        "market_condition": market_condition
    }

    # 2. Declara os módulos que contêm as definições de nós
    modules = [data_prep, feature_engineering]

    # 3. Inicializa os adaptadores (Tracking Hamilton UI, se ativado)
    adapters = []
    if enable_tracker:
        try:
            from hamilton_sdk import adapters as sdk_adapters
            # Tracker registra metadados, linhagem e telemetria no Hamilton UI
            tracker = sdk_adapters.HamiltonTracker(
                project_id=1,
                username="engenheiro_dados@empresa.com",
                dag_name=f"churn_pipeline_{market_condition}",
                tags={"environment": "production", "team": "data-science", "market_condition": market_condition},
                hamilton_api_url="http://localhost:8241",
                hamilton_ui_url="http://localhost:8242"
            )
            adapters.append(tracker)
            logger.info("Hamilton UI Tracker integrado com sucesso no Driver.")
        except ImportError:
            logger.warning(
                "Biblioteca 'hamilton-sdk' não encontrada. "
                "Para testar o tracker integrado com a UI, instale via: pip install 'apache-hamilton[ui,sdk]'"
            )

    # 4. Criação do Driver Hamilton
    # O driver analisa as dependências lógicas entre as funções dos módulos e gera a DAG
    dr = (
        driver.Builder()
        .with_config(config)
        .with_modules(*modules)
        .with_adapters(*adapters)
        .build()
    )

    # 5. Executa a DAG informando os nós finais que queremos obter
    # O Hamilton calcula dinamicamente apenas o caminho necessário para produzir os nós solicitados!
    target_nodes = ["final_feature_matrix", "mean_age"]
    logger.info(f"Executando pipeline para obter nós finais: {target_nodes} sob a condição '{market_condition}'...")

    results = dr.execute(target_nodes)

    final_df = results["final_feature_matrix"]
    mean_age_val = results["mean_age"]

    logger.info("=== RESULTADOS DA EXECUÇÃO ===")
    logger.info(f"Média de Idade Calculada (mean_age): {mean_age_val:.2f}")
    logger.info(f"Shape da Matriz de Features: {final_df.shape}")
    logger.info("\nPrimeiras linhas da Matriz de Features Gerada:")
    print(final_df.head(5).to_string())

    # Salva a matriz de features gerada localmente
    os.makedirs("output", exist_ok=True)
    try:
        parquet_path = "output/final_features.parquet"
        final_df.to_parquet(parquet_path, index=False)
        logger.info(f"Matriz de features exportada com sucesso para: {parquet_path}")
    except ImportError:
        csv_path = "output/final_features.csv"
        final_df.to_csv(csv_path, index=False)
        logger.info(f"Engine de Parquet não instalada. Matriz de features exportada resilientemente para CSV em: {csv_path}")

    # 6. Visualização Gráfica da DAG (Opcional)
    if visualize:
        try:
            logger.info("Renderizando grafo de execução e linhagem da DAG...")
            # Visualiza o grafo inteiro e salva a imagem correspondente
            dr.display_all_functions(output_file_path="output/hamilton_dag", format="png")
            logger.info("Grafo de Linhagem salvo em: output/hamilton_dag.png")
        except Exception as e:
            logger.error(
                f"Falha ao gerar visualização do grafo: {e}. "
                "Certifique-se de possuir a biblioteca 'graphviz' instalada no sistema operacional."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apache Hamilton Data Engineering & Feature Store Pipeline.")
    parser.add_argument(
        "--market_condition",
        choices=["stable", "high_inflation", "default"],
        default="stable",
        help="Ajusta o cálculo de cobranças mensais baseado no cenário de mercado (inflação alta ou estável)."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Flag para exportar visualização em PNG do grafo gerado."
    )
    parser.add_argument(
        "--enable_tracker",
        action="store_true",
        help="Ativa o Hamilton Tracker para logar as execuções na Hamilton UI."
    )
    args = parser.parse_args()

    run_pipeline(
        market_condition=args.market_condition,
        visualize=args.visualize,
        enable_tracker=args.enable_tracker
    )
