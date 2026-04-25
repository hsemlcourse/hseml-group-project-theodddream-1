[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — Прогнозирование суммы банковской транзакции клиента

**Студент:** Матвеев Егор Александрович

**Группа:** БИВ236


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Регрессия. Предсказание суммы банковской транзакции клиента (`TransactionAmount (INR)`) по демографическим данным (возраст, пол, локация) и текущему балансу счёта.

**Датасет:** [Bank Customer Segmentation (1M+ Transactions)](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation) — 1 048 567 строк, 9 колонок.

**Целевая метрика:** **MAE** (Mean Absolute Error, INR) — основная.
Дополнительно отслеживаем **RMSE** и **R²**.

Обоснование выбора:
- Распределение `TransactionAmount` крайне правоскошенное (skew ≈ 47, медиана ≈ 459 INR, среднее ≈ 1 574 INR, максимум ≈ 1 560 035 INR). При таком разбросе RMSE доминируется единичными крупными транзакциями и плохо отражает качество на типичной операции.
- **MAE** робастна к выбросам, измеряется в тех же единицах (INR) и имеет прозрачную бизнес-интерпретацию: «в среднем модель ошибается на X рупий».
- **RMSE** оставляем как вторичную метрику, чтобы контролировать поведение на крупных суммах (где ошибки штрафуются сильнее).
- **R²** — общая мера доли объяснённой дисперсии для сравнимости моделей.
- При сильно скошенном таргете дополнительно проверяем обучение на `log1p(TransactionAmount)` (см. `notebooks/03_experiments.ipynb`) — этот вариант штрафует относительную, а не абсолютную ошибку.

При выборе финальной модели приоритет — минимизация **MAE** на отложенной выборке.


## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные (gitignored)
│   └── raw                     # Исходный CSV из Kaggle (gitignored)
├── models                      # baseline.joblib, best_model.joblib (gitignored)
├── notebooks
│   ├── 01_eda.ipynb            # EDA на полном датасете, обоснование MAE
│   ├── 02_baseline.ipynb       # Baseline LinearRegression "из коробки"
│   └── 03_experiments.ipynb    # 5 моделей + тюнинг + ансамбль + PCA + log-target
├── presentation                # Презентация для защиты (на следующих чекпоинтах)
├── report
│   ├── images                  # Графики из EDA, попадают в report.md
│   └── report.md               # Финальный отчёт
├── src
│   ├── __init__.py             # SEED = 42
│   ├── preprocessing.py        # load_raw, clean, engineer_features, make_split, ...
│   └── modeling.py             # train_baseline, train_model, tune_*, build_ensemble, metrics
├── tests
│   └── test.py                 # Smoke-тесты пайплайна (pytest)
├── .github/workflows/ci.yml    # CI: ruff check src/
├── Dockerfile                  # python:3.10-slim + libgomp1 + requirements
├── pyproject.toml              # ruff и pytest конфиги
├── requirements.txt
└── README.md
```


## Запуск

### Локально (venv)

```bash
git clone https://github.com/hsemlcourse/hseml-group-project-TheOddDream.git
cd hseml-group-project-TheOddDream

# 1. Виртуальное окружение
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 2. Зависимости
pip install -r requirements.txt

# 3. Тесты и линтер
pytest -q tests/
ruff check src/ --line-length 120

# 4. Воспроизведение пайплайна (положите bank_transactions.csv в data/raw/ предварительно)
papermill notebooks/01_eda.ipynb        notebooks/01_eda.ipynb        --cwd notebooks
papermill notebooks/02_baseline.ipynb   notebooks/02_baseline.ipynb   --cwd notebooks
papermill notebooks/03_experiments.ipynb notebooks/03_experiments.ipynb --cwd notebooks
```

### Через Docker

```bash
docker build -t hseml-cp1 .
docker run --rm hseml-cp1                                  # дефолтный CMD: pytest -q tests/
docker run --rm hseml-cp1 ruff check src/ --line-length 120
# Запуск ноутбуков с подмонтированными данными:
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
    hseml-cp1 papermill notebooks/03_experiments.ipynb /tmp/out.ipynb --cwd notebooks
```


## Данные
- `data/raw/bank_transactions.csv` — исходный файл с Kaggle (~67 MB, 1 048 567 строк, 9 колонок). Не коммитим: на нём действует `.gitignore`.
- `data/processed/` — на CP1 не используется (вся предобработка делается на лету в `src/preprocessing.py`).


## Результаты

Сводная таблица по основным экспериментам (val/test получены на стратифицированных по `log1p(target)` отложенных выборках; финальная модель переобучена на полном train).

| Модель | MAE (INR) ↓ | RMSE (INR) ↓ | R² ↑ | Примечание |
|--------|-------------|--------------|------|------------|
| Baseline LinearRegression (val) | 1823.26 | 6274.13 | 0.00 | 2 сырые фичи, без feature engineering |
| Baseline LinearRegression (test) | 1820.12 | 6765.23 | 0.01 | |
| **LightGBM tuned, log1p(target)** (val) | **1346.99** | 6294.90 | ≈0.00 | Победитель по MAE; полный feature set |
| **LightGBM tuned, log1p(target)** (test) | **1343.92** | 6799.36 | ≈0.00 | Финальная модель, переобучена на полном train |

**Улучшение MAE относительно baseline:** −26% (с 1820 до 1344 INR).

**Замечание про R²/RMSE.** У победителя R² ≈ 0 и RMSE сопоставим с baseline — это ожидаемый эффект обучения на `log1p(target)`: модель оптимизирует относительную ошибку на типичных операциях (где плотность распределения максимальна) ценой худшей точности на гигантских транзакциях из правого хвоста. Это полностью согласовано с выбранной основной метрикой (MAE) и характером данных (skew ≈ 47).

Полный список экспериментов с гиперпараметрами и временем обучения — в `notebooks/03_experiments.ipynb` (раздел «Сводная таблица экспериментов»).


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
