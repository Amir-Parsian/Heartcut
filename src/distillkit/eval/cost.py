"""Cost model: dollars per 1,000 predictions for teachers and students.

* Teacher: logged API tokens x published price (cache reads and writes included), so the
  number is what the teacher costs as a classifier, call for call.
* Student: the hourly price of a reference CPU instance divided by the throughput
  measured on the benchmark machine. This assumes the instance is kept busy; at low
  traffic a serverless deployment would bill differently.
"""

from __future__ import annotations

SECONDS_PER_HOUR = 3600


def student_cost_per_1k(usd_per_hour: float, requests_per_second: float) -> float:
    if requests_per_second <= 0:
        raise ValueError("throughput must be positive")
    return usd_per_hour / (requests_per_second * SECONDS_PER_HOUR) * 1000


def teacher_cost_per_1k(costs_per_prediction: list[float]) -> float:
    if not costs_per_prediction:
        raise ValueError("no teacher costs")
    return sum(costs_per_prediction) / len(costs_per_prediction) * 1000


def blended_cost_per_1k(student: float, teacher: float, routed_share: float) -> float:
    """Every request hits the student; the routed share also pays for a teacher call."""
    return student + routed_share * teacher
