"""
Módulo de Preparação de Dados para o Pipeline Apache Hamilton.

Este módulo define transformações funcionais puras para a ingestão, tratamento
de valores ausentes (nulos) e preparação básica das colunas de dados de Churn.
Toda função aqui representa um nó da DAG do Hamilton, onde o nome da função
determina o nome de saída do nó na DAG (Function Name = Output Node Name).
"""

import pandas as pd
from typing import Dict, Any


def raw_data() -> pd.DataFrame:
    """
    Simula a ingestão de dados brutos de clientes (por exemplo, de um banco SQLite ou arquivo CSV).
    Retorna um DataFrame Pandas simulando dados de Churn de telecomunicações.
    """
    data = {
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age": [34, 45, None, 22, 58, 29, 41, None, 50, 31],
        "tenure": [4, 21, 1, 12, 72, 3, 48, 2, 60, 15],
        "monthly_charges": [65.5, 80.0, 20.1, None, 115.4, 55.0, 95.2, 45.0, None, 75.8],
        "total_charges": ["262.0", "1680.0", "20.1", "NaN", "8308.8", "165.0", "4569.6", "90.0", "NaN", "1137.0"],
        "internet_service": ["DSL", "Fiber", "No", "DSL", "Fiber", "Fiber", "DSL", "No", "Fiber", "DSL"],
        "churn": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    }
    return pd.DataFrame(data)


def customer_id(raw_data: pd.DataFrame) -> pd.Series:
    """Extrai a coluna customer_id do DataFrame de dados brutos."""
    return raw_data["customer_id"]


def age(raw_data: pd.DataFrame) -> pd.Series:
    """Extrai a coluna de idade dos dados brutos."""
    return raw_data["age"]


def tenure(raw_data: pd.DataFrame) -> pd.Series:
    """Extrai a coluna de tempo de casa (tenure) dos dados brutos."""
    return raw_data["tenure"]


def internet_service(raw_data: pd.DataFrame) -> pd.Series:
    """Extrai a coluna de tipo de serviço de internet."""
    return raw_data["internet_service"]


def churn(raw_data: pd.DataFrame) -> pd.Series:
    """Extrai a variável alvo (churn) dos dados brutos."""
    return raw_data["churn"]


def mean_age(age: pd.Series) -> float:
    """Calcula a média de idade para imputação posterior de nulos."""
    return float(age.mean())


def imputed_age(age: pd.Series, mean_age: float) -> pd.Series:
    """
    Tratamento de nulos: Imputa os valores ausentes de idade usando a média.
    """
    return age.fillna(mean_age)


def parsed_total_charges(raw_data: pd.DataFrame) -> pd.Series:
    """
    Tratamento de nulos/Conversão de tipo: Converte a coluna total_charges (que
    pode vir como string com representações de nulos) para float numérico,
    forçando erros a NaN.
    """
    return pd.to_numeric(raw_data["total_charges"], errors="coerce")


def imputed_total_charges(parsed_total_charges: pd.Series) -> pd.Series:
    """
    Trata valores ausentes na coluna de total_charges, preenchendo-os com 0.0.
    """
    return parsed_total_charges.fillna(0.0)


def raw_monthly_charges(raw_data: pd.DataFrame) -> pd.Series:
    """Extrai a coluna de cobranças mensais brutas (com possíveis nulos)."""
    return raw_data["monthly_charges"]


def imputed_monthly_charges(raw_monthly_charges: pd.Series) -> pd.Series:
    """
    Trata valores ausentes nas cobranças mensais, preenchendo com o valor mediano.
    """
    median_val = raw_monthly_charges.median()
    return raw_monthly_charges.fillna(median_val)


def normalized_monthly_charges(imputed_monthly_charges: pd.Series) -> pd.Series:
    """
    Normalização: Aplica Min-Max scaling na coluna de cobranças mensais.
    """
    min_val = imputed_monthly_charges.min()
    max_val = imputed_monthly_charges.max()
    if max_val == min_val:
        return imputed_monthly_charges * 0.0
    return (imputed_monthly_charges - min_val) / (max_val - min_val)


def base_prepared_df(
    customer_id: pd.Series,
    imputed_age: pd.Series,
    tenure: pd.Series,
    imputed_monthly_charges: pd.Series,
    imputed_total_charges: pd.Series,
    internet_service: pd.Series,
    churn: pd.Series
) -> pd.DataFrame:
    """
    Reúne todas as colunas pré-processadas e limpas em um único DataFrame base estruturado.
    """
    return pd.DataFrame({
        "customer_id": customer_id,
        "age": imputed_age,
        "tenure": tenure,
        "monthly_charges": imputed_monthly_charges,
        "total_charges": imputed_total_charges,
        "internet_service": internet_service,
        "churn": churn
    })
