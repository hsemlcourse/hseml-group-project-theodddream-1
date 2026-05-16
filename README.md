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
├── models                      # baseline.joblib, best_model_cp2.joblib (gitignored)
├── notebooks
│   ├── 01_eda.ipynb            # EDA на полном датасете, обоснование MAE
│   ├── 02_baseline.ipynb       # Baseline LinearRegression "из коробки"
│   ├── 03_experiments.ipynb    # CP1: 5 моделей + тюнинг + ансамбль + PCA + log-target
│   └── 04_cp2_experiments.ipynb # CP2: XGBoost, CatBoost, Stacking, Quantile, новый FE
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Графики из EDA и экспериментов
│   └── report.md               # Финальный отчёт (CP1 + CP2)
├── src
│   ├── __init__.py             # SEED = 42
│   ├── preprocessing.py        # load_raw, clean, engineer_features, make_split, ...
│   ├── modeling.py             # train_*, tune_*, build_stacking, build_ensemble, metrics
│   └── parser.py               # Парсинг данных: Kaggle API, ZIP, geo-enrichment
├── tests
│   └── test.py                 # 12 smoke-тестов пайплайна (pytest)
├── .github/workflows/ci.yml    # CI: ruff check src/
├── Dockerfile                  # python:3.10-slim + libgomp1 + requirements
├── docker-compose.yml          # Сервисы: app (тесты), lint (ruff), jupyter
├── pyproject.toml              # ruff и pytest конфиги
├── requirements.txt            # Пиннинг версий (numpy, pandas, sklearn, lgbm, xgb, catboost)
└── README.md
```


## Запуск

### Локально (venv)

```bash
git clone https://github.com/hsemlcourse/hseml-group-project-theodddream-1.git
cd hseml-group-project-theodddream-1

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

### Через Docker / Docker Compose

```bash
# Docker Compose (рекомендуется)
docker-compose up app          # запустить тесты
docker-compose up lint         # запустить линтер
docker-compose up jupyter      # Jupyter на порту 8888

# Или напрямую через Docker
docker build -t hseml-project .
docker run --rm hseml-project                                  # дефолтный CMD: pytest -q tests/
docker run --rm hseml-project ruff check src/ --line-length 120
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
    hseml-project papermill notebooks/04_cp2_experiments.ipynb /tmp/out.ipynb --cwd notebooks
```

### Парсинг данных

```bash
# Скачать с Kaggle API (нужен ~/.kaggle/kaggle.json)
python -m src.parser --download --validate

# Или извлечь из локального ZIP
python -m src.parser --extract --validate

# Получить гео-данные для обогащения
python -m src.parser --geo
```


## Данные
- `data/raw/bank_transactions.csv` — исходный файл с Kaggle (~67 MB, 1 048 567 строк, 9 колонок). Не коммитим: на нём действует `.gitignore`.
- `data/raw/india_cities_geo.csv` — гео-данные городов Индии (парсятся скриптом `src/parser.py`).
- `data/processed/` — вся предобработка делается на лету в `src/preprocessing.py`.


## Результаты

Сводная таблица (val/test на стратифицированных по `log1p(target)` отложенных выборках; финальная модель переобучена на полном train).

### CP1

| Модель | MAE (INR) | RMSE (INR) | R² | Примечание |
|--------|-----------|------------|-----|------------|
| Baseline LinearRegression (test) | 1820.12 | 6765.23 | 0.01 | 2 сырые фичи |
| LightGBM tuned, log1p(target) (test) | **1343.92** | 6799.36 | ~0.00 | CP1 победитель |

### CP2 (расширенный feature set + новые модели)

| Модель | MAE (INR) | Примечание |
|--------|-----------|------------|
| CatBoost (defaults) | 1912.29 | Лучший среди CP2 моделей |
| Stacking log1p(target) | 1922.29 | RF+LGBM+XGB → Ridge, log-шкала |
| CatBoost log1p(target) | 1922.30 | CatBoost + log-trick |
| XGBoost log1p(target) | 1924.87 | XGBoost + log-trick |
| LightGBM Quantile (alpha=0.5) | 1926.45 | Прямая минимизация MAE |
| Stacking (RF+LGBM+XGB → Ridge) | 1955.22 | Мета-обучение |

**Вывод CP2:** расширенный FE (customer aggregates, target encoding) не улучшил CP1 winner — при ~2 транзакциях на клиента агрегаты оказались слишком шумными. Финальная модель проекта остаётся **CP1 winner** (LightGBM tuned, log1p, test MAE = 1343.92).

**Замечание про R²/RMSE.** У победителя R² ~ 0 — это ожидаемый эффект `log1p(target)`: модель оптимизирует относительную ошибку на типичных операциях ценой точности на гигантских транзакциях. Согласовано с основной метрикой MAE и характером данных (skew ~ 47).

Полные таблицы экспериментов: `notebooks/03_experiments.ipynb` (CP1) и `notebooks/04_cp2_experiments.ipynb` (CP2).


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
