# 📂 Projetos End-to-End de Machine Learning

Bem-vindo à área de projetos práticos! Esta pasta é dedicada a notebooks que demonstram o ciclo completo de um projeto de Machine Learning (EDA, Pré-processamento, Modelagem e Avaliação), aplicando os algoritmos estudados na raiz do repositório.

## 🎯 Objetivo

Diferente dos notebooks de _tutoriais_ (focados em explicar um algoritmo isolado), os notebooks aqui devem focar na **resolução de um problema de negócio ou desafio de dados** do início ao fim.

## 📏 Padrão de Nomenclatura

Para manter a organização, seguimos estritamente o seguinte padrão de nomenclatura para os arquivos `.ipynb`:

`[ID].[SubID]-[nome_do_projeto_snake_case].ipynb`

### Exemplos:

- ✅ `1.1-bank_marketing.ipynb`
- ✅ `1.2-credit_risk_analysis.ipynb`
- ✅ `2.0-house_prices_prediction.ipynb`
- ❌ `analise_banco.ipynb` (Falta ID)
- ❌ `1.1-BankMarketing.ipynb` (Usar snake_case)

## 🤝 Como Contribuir

1.  **Crie uma Branch:** Nunca commite direto na `main`. Crie uma branch para o seu projeto:
    ```bash
    git checkout -b feat/projeto-credit-risk
    ```
2.  **Adicione seu Notebook:** Salve seu trabalho nesta pasta seguindo a nomenclatura acima.
3.  **Dados e Modelos:**
    - **NÃO** suba arquivos de dados pesados (`.csv`, `.zip`, `.parquet`) ou modelos binários (`.pkl`, `.h5`) se forem maiores que 10MB.
    - Se necessário, inclua um link para o dataset no topo do seu notebook ou instruções de como baixá-lo.
4.  **Pull Request:** Abra um PR descrevendo brevemente o problema que seu projeto resolve.

## ⚠️ Atenção

- Certifique-se de limpar as saídas (outputs) do notebook antes de commitar se elas conterem imagens muito pesadas ou dados sensíveis.
- Documente as bibliotecas necessárias no início do notebook ou atualize o `requirements.txt` na raiz se usar algo novo.

---
