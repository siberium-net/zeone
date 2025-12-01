"""
Metrics Collector - Сбор метрик
===============================

[METRICS] Типы метрик:
- Counter: монотонно возрастающий (requests, errors)
- Gauge: текущее значение (connections, memory)
- Histogram: распределение (latency, sizes)

[EXPORT] Форматы экспорта:
- Prometheus text format
- JSON
- StatsD

[LABELS] Поддержка labels для группировки
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Значение метрики."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """
    Counter метрика - монотонно возрастающая.
    
    [USAGE]
    ```python
    requests = Counter("requests_total", "Total requests")
    requests.inc()
    requests.inc(5)
    requests.inc(labels={"method": "GET"})
    ```
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        
        self._values: Dict[Tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Увеличить счётчик."""
        label_values = self._make_label_key(labels)
        with self._lock:
            self._values[label_values] += amount
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Получить значение."""
        label_values = self._make_label_key(labels)
        return self._values.get(label_values, 0.0)
    
    def get_all(self) -> List[MetricValue]:
        """Получить все значения."""
        result = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(zip(self.label_names, label_key)) if self.label_names else {}
                result.append(MetricValue(value=value, labels=labels))
        return result
    
    def _make_label_key(self, labels: Optional[Dict[str, str]]) -> Tuple:
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)


class Gauge:
    """
    Gauge метрика - текущее значение.
    
    [USAGE]
    ```python
    connections = Gauge("connections_active", "Active connections")
    connections.set(10)
    connections.inc()
    connections.dec()
    ```
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        
        self._values: Dict[Tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Установить значение."""
        label_values = self._make_label_key(labels)
        with self._lock:
            self._values[label_values] = value
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Увеличить."""
        label_values = self._make_label_key(labels)
        with self._lock:
            self._values[label_values] += amount
    
    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Уменьшить."""
        label_values = self._make_label_key(labels)
        with self._lock:
            self._values[label_values] -= amount
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Получить значение."""
        label_values = self._make_label_key(labels)
        return self._values.get(label_values, 0.0)
    
    def get_all(self) -> List[MetricValue]:
        """Получить все значения."""
        result = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(zip(self.label_names, label_key)) if self.label_names else {}
                result.append(MetricValue(value=value, labels=labels))
        return result
    
    def _make_label_key(self, labels: Optional[Dict[str, str]]) -> Tuple:
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)


class Histogram:
    """
    Histogram метрика - распределение значений.
    
    [USAGE]
    ```python
    latency = Histogram("request_latency", "Request latency",
                        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
    latency.observe(0.15)
    
    # Или с context manager
    with latency.time():
        do_something()
    ```
    """
    
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        
        self._bucket_counts: Dict[Tuple, Dict[float, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._sums: Dict[Tuple, float] = defaultdict(float)
        self._counts: Dict[Tuple, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Записать наблюдение."""
        label_values = self._make_label_key(labels)
        
        with self._lock:
            self._sums[label_values] += value
            self._counts[label_values] += 1
            
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[label_values][bucket] += 1
    
    def time(self, labels: Optional[Dict[str, str]] = None):
        """Context manager для измерения времени."""
        return _HistogramTimer(self, labels)
    
    def get_stats(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Получить статистику."""
        label_values = self._make_label_key(labels)
        
        with self._lock:
            count = self._counts.get(label_values, 0)
            total = self._sums.get(label_values, 0.0)
            
            return {
                "count": count,
                "sum": total,
                "avg": total / count if count > 0 else 0,
                "buckets": dict(self._bucket_counts.get(label_values, {})),
            }
    
    def _make_label_key(self, labels: Optional[Dict[str, str]]) -> Tuple:
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)


class _HistogramTimer:
    """Context manager для Histogram.time()."""
    
    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]]):
        self.histogram = histogram
        self.labels = labels
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.histogram.observe(elapsed, self.labels)


class MetricsCollector:
    """
    Централизованный сборщик метрик.
    
    [USAGE]
    ```python
    collector = MetricsCollector()
    
    # Регистрация метрик
    collector.counter("requests_total", "Total requests")
    collector.gauge("connections", "Active connections")
    collector.histogram("latency", "Request latency")
    
    # Использование
    collector.inc("requests_total")
    collector.set("connections", 42)
    collector.observe("latency", 0.15)
    
    # Экспорт
    print(collector.export_prometheus())
    ```
    """
    
    def __init__(self, prefix: str = "p2p"):
        """
        Args:
            prefix: Префикс для всех метрик
        """
        self.prefix = prefix
        
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        
        # Register default metrics
        self._register_defaults()
    
    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Создать Counter."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._counters:
            self._counters[full_name] = Counter(full_name, description, labels)
        return self._counters[full_name]
    
    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Создать Gauge."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._gauges:
            self._gauges[full_name] = Gauge(full_name, description, labels)
        return self._gauges[full_name]
    
    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """Создать Histogram."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._histograms:
            self._histograms[full_name] = Histogram(full_name, description, labels, buckets)
        return self._histograms[full_name]
    
    def inc(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Увеличить counter."""
        full_name = f"{self.prefix}_{name}"
        if full_name in self._counters:
            self._counters[full_name].inc(amount, labels)
    
    def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Установить gauge."""
        full_name = f"{self.prefix}_{name}"
        if full_name in self._gauges:
            self._gauges[full_name].set(value, labels)
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Записать в histogram."""
        full_name = f"{self.prefix}_{name}"
        if full_name in self._histograms:
            self._histograms[full_name].observe(value, labels)
    
    def export_prometheus(self) -> str:
        """Экспорт в Prometheus text format."""
        lines = []
        
        # Counters
        for name, counter in self._counters.items():
            if counter.description:
                lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            
            for mv in counter.get_all():
                label_str = self._format_labels(mv.labels)
                lines.append(f"{name}{label_str} {mv.value}")
        
        # Gauges
        for name, gauge in self._gauges.items():
            if gauge.description:
                lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            
            for mv in gauge.get_all():
                label_str = self._format_labels(mv.labels)
                lines.append(f"{name}{label_str} {mv.value}")
        
        # Histograms
        for name, histogram in self._histograms.items():
            if histogram.description:
                lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")
            
            # TODO: Proper histogram export with buckets
        
        return "\n".join(lines)
    
    def export_json(self) -> Dict[str, Any]:
        """Экспорт в JSON."""
        result = {
            "timestamp": time.time(),
            "counters": {},
            "gauges": {},
            "histograms": {},
        }
        
        for name, counter in self._counters.items():
            values = counter.get_all()
            if len(values) == 1 and not values[0].labels:
                result["counters"][name] = values[0].value
            else:
                result["counters"][name] = [
                    {"value": v.value, "labels": v.labels} for v in values
                ]
        
        for name, gauge in self._gauges.items():
            values = gauge.get_all()
            if len(values) == 1 and not values[0].labels:
                result["gauges"][name] = values[0].value
            else:
                result["gauges"][name] = [
                    {"value": v.value, "labels": v.labels} for v in values
                ]
        
        for name, histogram in self._histograms.items():
            result["histograms"][name] = histogram.get_stats()
        
        return result
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Форматировать labels для Prometheus."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(parts) + "}"
    
    def _register_defaults(self) -> None:
        """Зарегистрировать стандартные метрики."""
        # Network
        self.counter("messages_sent_total", "Total messages sent", ["type"])
        self.counter("messages_received_total", "Total messages received", ["type"])
        self.counter("bytes_sent_total", "Total bytes sent")
        self.counter("bytes_received_total", "Total bytes received")
        self.counter("connections_total", "Total connections", ["direction"])
        self.counter("connection_errors_total", "Connection errors", ["reason"])
        
        # State
        self.gauge("peers_connected", "Currently connected peers")
        self.gauge("peers_known", "Known peers")
        self.gauge("dht_stored_keys", "DHT stored keys")
        
        # Performance
        self.histogram("message_latency_seconds", "Message processing latency")
        self.histogram("connection_duration_seconds", "Connection duration")
        
        # Economy
        self.gauge("balance_total", "Total balance with all peers")
        self.counter("transactions_total", "Total transactions", ["type"])
        
        # Security
        self.counter("rate_limited_total", "Rate limited requests", ["action"])
        self.counter("banned_peers_total", "Banned peers")


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Получить глобальный MetricsCollector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics

