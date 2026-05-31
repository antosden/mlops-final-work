# SLI/SLO для MLOps-системы дедупликации профилей

## 1. Общая цель мониторинга

Система предназначена для поиска дубликатов профилей клиентов на маркетплейсе скидок.  
Мониторинг должен контролировать не только техническое состояние сервисов, но и качество ML-модели, так как ошибочное автоматическое объединение разных пользователей может привести к бизнес-рискам.

Основной приоритет системы — высокая точность автоматического объединения профилей.

---

## 2. Model Serving Component

Компонент: FastAPI-сервис для получения предсказаний модели.

### SLI

- Доступность API.
- Latency endpoint `/predict`.
- Доля ошибочных HTTP-ответов 5xx.
- Доля невалидных запросов 4xx.
- Количество prediction-запросов в минуту.

### SLO

- Доступность API: не ниже 99% в месяц.
- P95 latency endpoint `/predict`: не более 500 мс.
- Доля 5xx-ошибок: не более 1% запросов.
- Доля невалидных 4xx-запросов: не более 5% запросов.

### Риск нарушения

Если API недоступен или отвечает слишком медленно, downstream-сервисы не смогут получать рекомендации по объединению профилей.

---

## 3. Model Training Infrastructure

Компонент: скрипты подготовки данных и обучения моделей.

### SLI

- Успешность запуска `prepare_dataset.py`.
- Успешность запуска `train_baseline.py`.
- Успешность запуска `train_candidate.py`.
- Время выполнения пайплайна подготовки данных.
- Время обучения модели.
- Наличие сохранённого model artifact.

### SLO

- Успешность training pipeline: не ниже 95% запусков.
- Время подготовки датасета: не более 30 минут для учебного объёма данных.
- Время обучения модели: не более 30 минут для учебного объёма данных.
- После успешного обучения model artifact должен быть сохранён в `model_artifacts/`.

### Риск нарушения

Если training pipeline нестабилен, команда не сможет воспроизводимо обучать и сравнивать новые версии модели.

---

## 4. Feature Store / Offline Feature Layer

Компонент: offline feature datasets в parquet-формате.

В учебной реализации feature store представлен как offline feature layer:

- `offline_feature_store/raw/profiles.parquet`
- `offline_feature_store/features/train_pairs.parquet`
- `offline_feature_store/features/validation_pairs.parquet`
- `offline_feature_store/features/test_pairs.parquet`

### SLI

- Наличие исходного parquet-файла.
- Наличие train/validation/test feature datasets.
- Доля пропусков в ключевых признаках.
- Количество строк в train/validation/test.
- Баланс классов по `label`.

### SLO

- Все обязательные feature-файлы должны существовать после запуска `prepare_dataset.py`.
- Доля полностью пустых feature rows: 0%.
- Доля положительного класса должна быть ненулевой во всех выборках.
- Train, validation и test не должны пересекаться по `entity_id`.

### Риск нарушения

Если признаки сформированы некорректно, модель может обучиться на невалидных данных или получить data leakage.

---

## 5. ML Metadata Store

Компонент: MLflow Tracking.

### SLI

- Наличие experiment `profile-deduplication`.
- Наличие run для baseline-модели.
- Наличие run для candidate-модели.
- Наличие залогированных параметров модели.
- Наличие залогированных метрик качества.
- Наличие model artifact.

### SLO

- Каждый training run должен логировать параметры модели.
- Каждый training run должен логировать метрики `precision`, `recall`, `f1`, `roc_auc`.
- Каждый training run должен сохранять model artifact.
- Для каждой candidate-модели должен быть сохранён результат quality gate.

### Риск нарушения

Если metadata не сохраняется, невозможно воспроизвести эксперимент и объяснить, почему модель была принята или отклонена.

---

## 6. Model Registry

Компонент: MLflow Model Registry.

### SLI

- Наличие зарегистрированной baseline-модели.
- Наличие зарегистрированной candidate-модели.
- Наличие production alias у принятой модели.
- Наличие тегов model version.

### SLO

- Candidate-модель должна попадать в registry после успешного обучения.
- Production alias может быть назначен только после прохождения quality gate.
- У production-модели должны быть сохранены теги:
  - `deployment_status=production`
  - `promoted_by=quality_gate`

### Риск нарушения

Если registry не контролируется, в production может попасть модель без проверки качества.

---

## 7. CI/CD Component

Компонент: GitLab CI.

### SLI

- Статус pipeline.
- Успешность установки зависимостей.
- Успешность проверки Python-кода.
- Успешность тестов.
- Успешность MLOps pipeline stages.

### SLO

- Pipeline должен успешно проходить на основной ветке.
- Проверка импортов и синтаксиса должна выполняться при каждом push.
- MLOps-скрипты должны запускаться при наличии обучающего датасета.
- Артефакты pipeline должны сохраняться как GitLab artifacts.

### Риск нарушения

Если CI/CD не работает, изменения в коде могут ломать воспроизводимость ML-пайплайна.

---

## 8. Model Quality Monitoring

Компонент: мониторинг качества модели.

### SLI

- Validation Precision.
- Validation Recall.
- Validation F1.
- Validation ROC-AUC.
- Test Precision.
- Test Recall.
- Test F1.
- Test ROC-AUC.
- Доля `auto_merge`.
- Доля `manual_review`.
- Доля `no_duplicate`.

### SLO

- Validation Precision: не ниже 0.90.
- Candidate Validation F1: не ниже baseline Validation F1.
- Новая модель не может быть переведена в production, если не прошла quality gate.
- Резкое изменение распределения `match_score` должно рассматриваться как сигнал возможного data drift.

### Риск нарушения

Если качество модели падает, система может ошибочно объединять разные профили или пропускать значимую долю дублей.

---

## 9. Итоговая таблица SLI/SLO

| Компонент | Основные SLI | Основные SLO |
|---|---|---|
| Model Serving | availability, latency, error rate | availability >= 99%, P95 latency <= 500 ms |
| Training Infrastructure | success rate, training duration, artifact exists | success rate >= 95%, artifact saved |
| Feature Store | feature files exist, class balance, missing values | no empty datasets, no entity leakage |
| ML Metadata Store | runs, params, metrics, artifacts | every run logs params, metrics and artifact |
| Model Registry | registered versions, aliases, tags | production alias only after quality gate |
| CI/CD | pipeline status, tests, syntax check | pipeline passes on main branch |
| Model Quality | precision, recall, F1, ROC-AUC | precision >= 0.90, F1 >= baseline |