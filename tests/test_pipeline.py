"""
Suíte de Testes Unitários e de Integração para o Pipeline Hamilton.

Este módulo demonstra como a arquitetura do Hamilton facilita testes de duas formas:
1. Teste Unitário direto de funções puras Python (sem instanciar o grafo, sem mocks complexos).
2. Teste de Integração executando a DAG do Hamilton de ponta a ponta.
"""

import pytest
import pandas as pd
import numpy as np

from hamilton import driver
from features import data_prep, feature_engineering


# =====================================================================
# 1. TESTES UNITÁRIOS DE FUNÇÕES PURAS (Unit Tests)
# =====================================================================

def test_unit_is_senior():
    """
    Testa individualmente a função pura `is_senior` com dados controlados.
    Não há necessidade de construir a DAG inteira para testar a lógica!
    """
    imputed_age_input = pd.Series([20, 59, 60, 85, 30])
    expected_output = pd.Series([0, 0, 1, 1, 0])

    output = feature_engineering.is_senior(imputed_age_input)

    pd.testing.assert_series_equal(output, expected_output)


def test_unit_tenure_years():
    """
    Testa a conversão de meses para anos na função pura `tenure_years`.
    """
    tenure_months = pd.Series([0, 12, 24, 36])
    expected_years = pd.Series([0.0, 1.0, 2.0, 3.0])

    output = feature_engineering.tenure_years(tenure_months)

    pd.testing.assert_series_equal(output, expected_years)


def test_unit_average_monthly_spend():
    """
    Testa o cálculo da média mensal de gastos na função pura `average_monthly_spend`,
    garantindo que divisões por zero sejam tratadas corretamente (retornando 0.0).
    """
    total_charges = pd.Series([100.0, 240.0, 0.0, 50.0])
    tenure_months = pd.Series([10, 12, 0, 0])

    expected_output = pd.Series([10.0, 20.0, 0.0, 0.0])

    output = feature_engineering.average_monthly_spend(total_charges, tenure_months)

    pd.testing.assert_series_equal(output, expected_output)


def test_unit_imputed_age():
    """
    Testa a imputação de nulos para idade usando a média fornecida.
    """
    age_series = pd.Series([20.0, None, 40.0])
    mean_val = 30.0
    expected_output = pd.Series([20.0, 30.0, 40.0])

    output = data_prep.imputed_age(age_series, mean_val)

    pd.testing.assert_series_equal(output, expected_output)


# =====================================================================
# 2. TESTES DE INTEGRAÇÃO DA DAG (Integration Tests)
# =====================================================================

@pytest.mark.parametrize("market_condition,expected_multiplier", [
    ("stable", 1.0),
    ("high_inflation", 1.10),
    ("default", 1.05)
])
def test_integration_hamilton_dag(market_condition, expected_multiplier):
    """
    Executa a DAG completa do Hamilton de ponta a ponta.
    Verifica se a ramificação de condições (@config.when) altera o valor final de 'adjusted_charges'
    conforme configurado, e se o schema validado pelo @check_output é produzido corretamente.
    """
    # Define as configurações iniciais
    config = {
        "market_condition": market_condition
    }

    # Instancia o Driver Hamilton carregando os dois módulos funcionais
    dr = (
        driver.Builder()
        .with_config(config)
        .with_modules(data_prep, feature_engineering)
        .build()
    )

    # Nós finais que desejamos extrair do grafo
    target_nodes = ["final_feature_matrix", "mean_age", "adjusted_monthly_charges"]

    results = dr.execute(target_nodes)

    # Verifica a existência dos outputs desejados
    assert "final_feature_matrix" in results
    assert "mean_age" in results
    assert "adjusted_monthly_charges" in results

    final_df = results["final_feature_matrix"]
    adjusted_charges = results["adjusted_monthly_charges"]
    imputed_monthly = final_df["monthly_charges"]

    # Verifica se a regra de negócio condicional foi executada corretamente
    # adjusted_charges deve ser igual a monthly_charges multiplicado pela taxa de inflação correspondente
    np.testing.assert_allclose(adjusted_charges, imputed_monthly * expected_multiplier, rtol=1e-5)

    # Verifica que o DataFrame final possui os tipos corretos e não tem valores NaN nos nós essenciais
    assert len(final_df) == 10
    assert not final_df["age"].isnull().any()
    assert not final_df["total_charges"].isnull().any()
    assert not final_df["avg_monthly_spend"].isnull().any()

    # Verifica que One-Hot Encoding gerou as colunas corretas binárias
    for col in ["is_dsl", "is_fiber", "is_no_internet"]:
        assert col in final_df.columns
        assert set(final_df[col].unique()).issubset({0, 1})
