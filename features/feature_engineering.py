"""
Módulo de Feature Engineering Avançado do Apache Hamilton.

Este módulo demonstra o uso de decoradores avançados do Hamilton, tais como:
- @config.when: ramificação condicional com base em configurações passadas ao Driver.
- @extract_columns: divide um DataFrame retornado em várias séries individuais.
- @check_output_custom: validação personalizada de qualidade de dados em tempo de execução.
- @tag: anotação e catalogação de metadados no grafo para fácil governança e auditoria.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np
from hamilton.function_modifiers import config, extract_columns, check_output_custom, tag
from hamilton.data_quality.base import DataValidator, ValidationResult


# =====================================================================
# VALIDADOR DE QUALIDADE DE DADOS PERSONALIZADO (Custom Schema Validator)
# =====================================================================

class DataFrameSchemaValidator(DataValidator):
    """
    Validador personalizado para garantir que um DataFrame do Pandas esteja em
    conformidade com os tipos de dados e nomes de colunas esperados.
    """
    def __init__(self, schema: Dict[str, Any], importance: str = "fail"):
        super().__init__(importance)
        self.schema = schema

    def applies_to(self, datatype: type) -> bool:
        """Indica se este validador se aplica ao tipo de dado de saída."""
        return issubclass(datatype, pd.DataFrame)

    def description(self) -> str:
        """Descrição inteligível do validador para relatórios de linhagem."""
        return "Valida se as colunas e os tipos de dados do Pandas DataFrame correspondem ao schema definido."

    @classmethod
    def name(cls) -> str:
        return "dataframe_schema_validator"

    def validate(self, dataset: Any) -> ValidationResult:
        """Executa a validação de schema e tipo."""
        if not isinstance(dataset, pd.DataFrame):
            return ValidationResult(
                passes=False,
                message=f"Tipo de dado inválido. Esperava pd.DataFrame, obteve: {type(dataset)}"
            )

        for col, expected_type in self.schema.items():
            # 1. Verifica existência da coluna
            if col not in dataset.columns:
                return ValidationResult(
                    passes=False,
                    message=f"Falha de Qualidade de Dados: Coluna requerida '{col}' está ausente no DataFrame."
                )

            # 2. Verifica compatibilidade de tipo
            actual_dtype = dataset[col].dtype
            if expected_type in (np.integer, int):
                if not pd.api.types.is_integer_dtype(actual_dtype):
                    return ValidationResult(
                        passes=False,
                        message=f"Falha de Tipo: Coluna '{col}' possui tipo '{actual_dtype}', esperava inteiro."
                    )
            elif expected_type in (np.floating, float):
                if not pd.api.types.is_float_dtype(actual_dtype):
                    return ValidationResult(
                        passes=False,
                        message=f"Falha de Tipo: Coluna '{col}' possui tipo '{actual_dtype}', esperava float."
                    )
            elif expected_type in (bool, np.bool_):
                if not pd.api.types.is_bool_dtype(actual_dtype):
                    return ValidationResult(
                        passes=False,
                        message=f"Falha de Tipo: Coluna '{col}' possui tipo '{actual_dtype}', esperava booleano."
                    )

        return ValidationResult(
            passes=True,
            message="DataFrame perfeitamente validado contra o Schema de Produção."
        )


# =====================================================================
# DEFINIÇÃO DAS TRANSFORMAÇÕES DO HAMILTON (Features Nodes)
# =====================================================================

@tag(feature_category="demographics")
def is_senior(imputed_age: pd.Series) -> pd.Series:
    """
    Cria uma variável binária indicando se o cliente é idoso (idade >= 60).
    """
    return (imputed_age >= 60).astype(int)


@tag(feature_category="engagement")
def tenure_years(tenure: pd.Series) -> pd.Series:
    """
    Converte o tempo de tenure de meses para anos.
    """
    return tenure / 12.0


@tag(feature_category="financials")
def average_monthly_spend(imputed_total_charges: pd.Series, tenure: pd.Series) -> pd.Series:
    """
    Calcula a média real gasta por mês dividindo os gastos totais pelo tenure.
    Para clientes com tenure zero, preenche com 0.
    """
    # Evita divisão por zero retornando 0.0 onde tenure == 0
    spend = imputed_total_charges / tenure.replace(0, np.nan)
    return spend.fillna(0.0)


@tag(feature_category="enrichment")
@config.when(market_condition="high_inflation")
def adjusted_monthly_charges__high(imputed_monthly_charges: pd.Series) -> pd.Series:
    """
    Calcula as taxas mensais ajustadas se a condição do mercado for de inflação alta (+10%).
    """
    return imputed_monthly_charges * 1.10


@tag(feature_category="enrichment")
@config.when(market_condition="stable")
def adjusted_monthly_charges__stable(imputed_monthly_charges: pd.Series) -> pd.Series:
    """
    Calcula as taxas mensais ajustadas se a condição do mercado for estável (sem ajuste).
    """
    return imputed_monthly_charges * 1.0


@tag(feature_category="enrichment")
@config.when(market_condition="default")
def adjusted_monthly_charges__default(imputed_monthly_charges: pd.Series) -> pd.Series:
    """
    Ajuste default para cobranças mensais (+5%).
    """
    return imputed_monthly_charges * 1.05


@tag(feature_category="encoding")
@extract_columns("is_dsl", "is_fiber", "is_no_internet")
def internet_service_dummies(internet_service: pd.Series) -> pd.DataFrame:
    """
    Gera One-Hot Encoding para a variável categórica 'internet_service'
    e extrai cada uma como uma coluna independente na DAG.
    """
    dummies = pd.get_dummies(internet_service, prefix="service")
    # Garante que as três colunas esperadas existam com nomes padronizados
    return pd.DataFrame({
        "is_dsl": dummies.get("service_DSL", pd.Series(0, index=internet_service.index)).astype(int),
        "is_fiber": dummies.get("service_Fiber", pd.Series(0, index=internet_service.index)).astype(int),
        "is_no_internet": dummies.get("service_No", pd.Series(0, index=internet_service.index)).astype(int)
    })


# Definição do schema esperado na matriz final de features
EXPECTED_SCHEMA = {
    "customer_id": np.integer,
    "age": np.floating,
    "is_senior": np.integer,
    "tenure": np.integer,
    "tenure_years": np.floating,
    "monthly_charges": np.floating,
    "adjusted_charges": np.floating,
    "total_charges": np.floating,
    "avg_monthly_spend": np.floating,
    "is_dsl": np.integer,
    "is_fiber": np.integer,
    "is_no_internet": np.integer,
    "churn": np.integer
}


@tag(feature_store="churn_feature_set")
@check_output_custom(DataFrameSchemaValidator(EXPECTED_SCHEMA, importance="fail"))
def final_feature_matrix(
    customer_id: pd.Series,
    imputed_age: pd.Series,
    is_senior: pd.Series,
    tenure: pd.Series,
    tenure_years: pd.Series,
    imputed_monthly_charges: pd.Series,
    adjusted_monthly_charges: pd.Series,  # O dunderscore '__...' em adjusted_monthly_charges__... resolve para este nome
    imputed_total_charges: pd.Series,
    average_monthly_spend: pd.Series,
    is_dsl: pd.Series,
    is_fiber: pd.Series,
    is_no_internet: pd.Series,
    churn: pd.Series
) -> pd.DataFrame:
    """
    Agrupa todas as transformações de feature engineering e limpeza em uma matriz de
    features consolidada pronta para treinamento de modelos de Machine Learning.
    Este nó é protegido pelo validador customizado `@check_output_custom` para garantir qualidade.
    """
    return pd.DataFrame({
        "customer_id": customer_id.astype(int),
        "age": imputed_age.astype(float),
        "is_senior": is_senior.astype(int),
        "tenure": tenure.astype(int),
        "tenure_years": tenure_years.astype(float),
        "monthly_charges": imputed_monthly_charges.astype(float),
        "adjusted_charges": adjusted_monthly_charges.astype(float),
        "total_charges": imputed_total_charges.astype(float),
        "avg_monthly_spend": average_monthly_spend.astype(float),
        "is_dsl": is_dsl.astype(int),
        "is_fiber": is_fiber.astype(int),
        "is_no_internet": is_no_internet.astype(int),
        "churn": churn.astype(int)
    })
