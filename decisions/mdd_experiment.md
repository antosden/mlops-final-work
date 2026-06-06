## Шаг 5. Принятие решений по Metrics Driven Development

### 1. Цель MDD-анализа

Цель анализа — проверить, действительно ли улучшенная архитектура системы снижает latency backend-сервиса.

В рамках задания сравниваются два набора данных:

- `existing_system_responses` — latency существующей системы;
- `improved_system_responses` — latency улучшенной системы.

Решение принимается не визуально, а на основе статистического теста.

---

### 2. Метрика

Основная метрика:

- latency, seconds.

Дополнительно анализируются:

- mean latency;
- p95 latency;
- p99 latency.

Latency выбрана как техническая SLI, потому что backend-сервис должен быстро отвечать на запросы пользователей и других систем.

---

### 3. Визуализация распределений

Для первичного анализа были построены KDE-графики распределений latency для существующей и улучшенной системы.

По графику видно, что распределение улучшенной системы смещено влево, то есть новая система в среднем отвечает быстрее.

Однако визуального сравнения недостаточно, поэтому далее была проведена статистическая проверка гипотез.

---

### 4. Гипотезы

H0: среднее время отклика улучшенной системы не меньше среднего времени отклика существующей системы.

Формально:

H0: mean_improved >= mean_existing

H1: среднее время отклика улучшенной системы меньше среднего времени отклика существующей системы.

Формально:

H1: mean_improved < mean_existing

---

### 5. Уровень значимости

Для проверки гипотез выбран уровень значимости:

alpha = 0.05

Это означает, что мы допускаем вероятность ошибки первого рода 5%.

---

### 6. Статистический тест

Для сравнения двух независимых выборок latency был выбран Welch's t-test.

Причины выбора:

- сравниваются две независимые выборки;
- анализируется различие средних значений latency;
- Welch's t-test не требует равенства дисперсий.

---

### 7. Код статистического анализа

```python
import numpy as np
from scipy import stats

np.random.seed(42)

existing_system_responses = np.random.normal(loc=3.5, scale=0.4, size=500000)
improved_system_responses = np.random.normal(loc=2.0, scale=0.4, size=500000)

existing_mean = np.mean(existing_system_responses)
improved_mean = np.mean(improved_system_responses)

existing_p95 = np.percentile(existing_system_responses, 95)
improved_p95 = np.percentile(improved_system_responses, 95)

existing_p99 = np.percentile(existing_system_responses, 99)
improved_p99 = np.percentile(improved_system_responses, 99)

t_stat, p_two_sided = stats.ttest_ind(
    improved_system_responses,
    existing_system_responses,
    equal_var=False
)

p_value = p_two_sided / 2
alpha = 0.05

latency_reduction = (existing_mean - improved_mean) / existing_mean * 100

print("Existing mean latency:", round(existing_mean, 4))
print("Improved mean latency:", round(improved_mean, 4))

print("Existing p95 latency:", round(existing_p95, 4))
print("Improved p95 latency:", round(improved_p95, 4))

print("Existing p99 latency:", round(existing_p99, 4))
print("Improved p99 latency:", round(improved_p99, 4))

print("t-statistic:", t_stat)
print("one-sided p-value:", p_value)
print("Latency reduction:", round(latency_reduction, 2), "%")

if p_value < alpha and improved_mean < existing_mean:
    print("H0 rejected: improved system has statistically lower latency.")
else:
    print("H0 not rejected: no statistically significant latency improvement.")
