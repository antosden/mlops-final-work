# MDD-эксперимент: сравнение baseline и candidate модели

## 1. Цель эксперимента

Цель эксперимента — проверить, можно ли заменить baseline-модель на candidate-модель в системе дедупликации профилей клиентов.

Система решает задачу бинарной классификации пары профилей:

- `1` — профили принадлежат одному пользователю;
- `0` — профили принадлежат разным пользователям.

Основной бизнес-риск — ошибочное автоматическое объединение разных пользователей. Поэтому ключевой метрикой является Precision.

---

## 2. Гипотеза

Candidate-модель LightGBM должна показывать качество не хуже baseline-модели CatBoost и проходить минимальный порог Precision.

Гипотеза:

```text
LightGBM candidate может быть переведён в production,
если validation_precision >= 0.90
и validation_f1 >= baseline_validation_f1.
```

## 3. Данные

Для эксперимента использовался датасет пар профилей.

Исходные данные:

offline_feature_store/raw/profiles.parquet

После подготовки данных формируются feature datasets:

offline_feature_store/features/train_pairs.parquet
offline_feature_store/features/validation_pairs.parquet
offline_feature_store/features/test_pairs.parquet

Разбиение выполняется по entity_id, чтобы один и тот же реальный пользователь не попадал одновременно в train, validation и test. Это снижает риск data leakage.

## 4. Модели

В эксперименте сравниваются две модели.

Baseline model
CatBoostClassifier

Роль baseline-модели — дать начальную точку сравнения качества.

Candidate model
LightGBMClassifier

Candidate-модель рассматривается как новая версия модели, которую можно перевести в production только после прохождения quality gate.

## 5. Метрики

Для оценки качества использовались offline-метрики:

Precision;
Recall;
F1-score;
ROC-AUC.

Основные метрики для принятия решения:

validation_precision
validation_f1

Дополнительно анализировались:

test_precision
test_recall
test_f1
test_roc_auc

## 6. Quality Gate

Candidate-модель допускается к promotion только при выполнении условий:

validation_precision >= 0.90
candidate_validation_f1 >= baseline_validation_f1

Если хотя бы одно условие не выполнено, candidate-модель отклоняется.

## 7. Результат эксперимента

По результатам запуска MLOps-пайплайна candidate-модель LightGBM показала метрики выше baseline-модели на test-наборе и прошла quality gate.

Решение:

PROMOTE_CANDIDATE

Candidate-модель была зарегистрирована в MLflow Model Registry и получила alias:

production

## 8. Интерпретация результата

Результат показывает, что новая модель может быть использована вместо baseline, так как она:

достигает минимального порога Precision;
не ухудшает F1 относительно baseline;
логируется и воспроизводится через MLflow;
проходит формальный quality gate перед promotion.

Такой подход снижает риск случайного выката модели с худшим качеством.

## 9. Ограничения эксперимента

Ограничения текущего эксперимента:

оценка выполнена offline;
нет live feedback от ручной проверки дублей;
нет production data drift monitoring;
порог 0.90 выбран как учебный бизнес-порог и может быть пересмотрен после накопления реальных данных;
тестовая выборка отражает текущую структуру датасета, но не гарантирует устойчивость на будущих данных.

## 10. Вывод

MDD-подход позволил принять решение о promotion модели на основе измеримых метрик, а не вручную.

В рамках учебной MLOps-системы реализован следующий цикл:

baseline training;
candidate training;
metrics logging;
quality gate;
promotion decision;
production alias in MLflow.

Это закрывает ключевое требование управляемого жизненного цикла ML-модели.