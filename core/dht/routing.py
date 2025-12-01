"""
Kademlia Routing Table
======================

[KADEMLIA] K-bucket таблица маршрутизации:
- XOR-метрика расстояния между node_id
- 160 k-buckets (по битам расстояния)
- k = 20 узлов на bucket (стандарт Kademlia)
- LRU-замена: новые узлы вытесняют неактивные

[XOR] Почему XOR:
- XOR(a, a) = 0 (узел ближе всего к себе)
- XOR(a, b) = XOR(b, a) (симметрия)
- XOR(a, b) + XOR(b, c) >= XOR(a, c) (неравенство треугольника)
- Унарность: для любого a, b существует ровно один c: XOR(a, c) = b
"""

import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Iterator
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


# Константы Kademlia
K = 20  # Размер k-bucket (количество узлов)
ALPHA = 3  # Параллельность запросов
ID_BITS = 160  # Битность идентификаторов (SHA-1)
BUCKET_REFRESH_INTERVAL = 3600  # Секунды между обновлениями bucket


@dataclass
class NodeInfo:
    """
    Информация об узле в DHT.
    
    [KADEMLIA] Каждый узел идентифицируется:
    - node_id: 160-битный идентификатор (SHA-1 от публичного ключа)
    - host/port: сетевой адрес
    - last_seen: время последнего контакта (для LRU)
    """
    
    node_id: bytes  # 20 байт (160 бит)
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)
    failed_requests: int = 0
    
    @property
    def node_id_hex(self) -> str:
        """Node ID в hex формате."""
        return self.node_id.hex()
    
    @property
    def address(self) -> Tuple[str, int]:
        """Адрес как кортеж (host, port)."""
        return (self.host, self.port)
    
    def touch(self) -> None:
        """Обновить время последнего контакта."""
        self.last_seen = time.time()
        self.failed_requests = 0
    
    def mark_failed(self) -> None:
        """Отметить неудачный запрос."""
        self.failed_requests += 1
    
    def is_stale(self, timeout: float = 900) -> bool:
        """Проверить, устарел ли узел (15 минут по умолчанию)."""
        return time.time() - self.last_seen > timeout
    
    def to_dict(self) -> Dict:
        """Сериализация в словарь."""
        return {
            "node_id": self.node_id.hex(),
            "host": self.host,
            "port": self.port,
            "last_seen": self.last_seen,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "NodeInfo":
        """Десериализация из словаря."""
        return cls(
            node_id=bytes.fromhex(data["node_id"]),
            host=data["host"],
            port=data["port"],
            last_seen=data.get("last_seen", time.time()),
        )
    
    def __hash__(self) -> int:
        return hash(self.node_id)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, NodeInfo):
            return self.node_id == other.node_id
        return False


def xor_distance(id1: bytes, id2: bytes) -> int:
    """
    Вычислить XOR-расстояние между двумя идентификаторами.
    
    [KADEMLIA] XOR-метрика:
    - Расстояние = побитовый XOR двух ID как целое число
    - Чем меньше значение, тем ближе узлы
    
    Args:
        id1: Первый идентификатор (20 байт)
        id2: Второй идентификатор (20 байт)
    
    Returns:
        XOR-расстояние как целое число
    """
    if len(id1) != len(id2):
        raise ValueError(f"ID length mismatch: {len(id1)} vs {len(id2)}")
    
    # XOR каждого байта и преобразуем в целое число
    xor_bytes = bytes(a ^ b for a, b in zip(id1, id2))
    return int.from_bytes(xor_bytes, byteorder='big')


def distance_to_bucket_index(distance: int) -> int:
    """
    Определить индекс bucket по XOR-расстоянию.
    
    [KADEMLIA] Bucket index = позиция старшего установленного бита:
    - Расстояние 0 -> bucket 0 (невозможно, это мы сами)
    - Расстояние 1 -> bucket 0
    - Расстояние 2-3 -> bucket 1
    - Расстояние 4-7 -> bucket 2
    - ...
    - Расстояние 2^159 - 2^160-1 -> bucket 159
    
    Args:
        distance: XOR-расстояние
    
    Returns:
        Индекс bucket (0-159)
    """
    if distance == 0:
        return 0
    
    # Находим позицию старшего бита (bit_length - 1)
    return distance.bit_length() - 1


def node_id_to_bucket_index(local_id: bytes, remote_id: bytes) -> int:
    """
    Определить индекс bucket для удалённого узла.
    
    Args:
        local_id: Наш node_id
        remote_id: Node_id удалённого узла
    
    Returns:
        Индекс bucket (0-159)
    """
    distance = xor_distance(local_id, remote_id)
    return distance_to_bucket_index(distance)


def generate_random_id_in_bucket(local_id: bytes, bucket_index: int) -> bytes:
    """
    Сгенерировать случайный ID, который попадёт в указанный bucket.
    
    [KADEMLIA] Используется для обновления bucket:
    - Генерируем ID с нужным расстоянием от local_id
    - Затем ищем узлы близкие к этому ID
    
    Args:
        local_id: Наш node_id
        bucket_index: Целевой bucket (0-159)
    
    Returns:
        Случайный node_id для lookup
    """
    import os
    
    # Генерируем случайные байты
    random_bytes = bytearray(os.urandom(20))
    
    # Устанавливаем бит в позиции bucket_index
    # и сбрасываем все биты выше
    byte_index = (ID_BITS - 1 - bucket_index) // 8
    bit_index = bucket_index % 8
    
    # Сбрасываем все биты выше bucket_index
    for i in range(byte_index):
        random_bytes[i] = local_id[i]
    
    # XOR с local_id чтобы получить нужное расстояние
    result = bytes(a ^ b for a, b in zip(random_bytes, local_id))
    return result


class KBucket:
    """
    K-bucket для хранения узлов с определённым расстоянием.
    
    [KADEMLIA] Каждый bucket:
    - Хранит до k узлов
    - Узлы упорядочены по времени последнего контакта (LRU)
    - Новые узлы добавляются в конец
    - При переполнении: проверяем head, если жив - отбрасываем новый
    """
    
    def __init__(self, k: int = K):
        """
        Args:
            k: Максимальный размер bucket
        """
        self.k = k
        # OrderedDict для LRU: ключ = node_id, значение = NodeInfo
        self._nodes: OrderedDict[bytes, NodeInfo] = OrderedDict()
        self._lock = threading.Lock()
        self.last_updated = time.time()
    
    def __len__(self) -> int:
        return len(self._nodes)
    
    def __iter__(self) -> Iterator[NodeInfo]:
        with self._lock:
            return iter(list(self._nodes.values()))
    
    def __contains__(self, node_id: bytes) -> bool:
        return node_id in self._nodes
    
    @property
    def is_full(self) -> bool:
        """Проверить, заполнен ли bucket."""
        return len(self._nodes) >= self.k
    
    @property
    def nodes(self) -> List[NodeInfo]:
        """Получить список узлов."""
        with self._lock:
            return list(self._nodes.values())
    
    @property
    def head(self) -> Optional[NodeInfo]:
        """Получить самый старый узел (head of LRU)."""
        with self._lock:
            if self._nodes:
                return next(iter(self._nodes.values()))
            return None
    
    @property
    def tail(self) -> Optional[NodeInfo]:
        """Получить самый новый узел (tail of LRU)."""
        with self._lock:
            if self._nodes:
                return list(self._nodes.values())[-1]
            return None
    
    def get(self, node_id: bytes) -> Optional[NodeInfo]:
        """Получить узел по ID."""
        return self._nodes.get(node_id)
    
    def add(self, node: NodeInfo) -> Tuple[bool, Optional[NodeInfo]]:
        """
        Добавить узел в bucket.
        
        [KADEMLIA] Логика добавления:
        1. Если узел уже есть -> обновляем и перемещаем в конец
        2. Если bucket не полон -> добавляем в конец
        3. Если bucket полон -> возвращаем head для проверки
        
        Returns:
            (added, eviction_candidate):
            - (True, None) - узел добавлен
            - (False, NodeInfo) - bucket полон, нужно проверить этот узел
            - (False, None) - ошибка
        """
        with self._lock:
            # Узел уже есть - обновляем
            if node.node_id in self._nodes:
                self._nodes.move_to_end(node.node_id)
                self._nodes[node.node_id].touch()
                self.last_updated = time.time()
                return True, None
            
            # Bucket не полон - добавляем
            if not self.is_full:
                self._nodes[node.node_id] = node
                self.last_updated = time.time()
                return True, None
            
            # Bucket полон - возвращаем head для проверки
            head = next(iter(self._nodes.values()))
            return False, head
    
    def remove(self, node_id: bytes) -> bool:
        """
        Удалить узел из bucket.
        
        Returns:
            True если узел был удалён
        """
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                return True
            return False
    
    def touch(self, node_id: bytes) -> bool:
        """
        Обновить время контакта и переместить в конец LRU.
        
        Returns:
            True если узел найден и обновлён
        """
        with self._lock:
            if node_id in self._nodes:
                self._nodes.move_to_end(node_id)
                self._nodes[node_id].touch()
                self.last_updated = time.time()
                return True
            return False
    
    def replace_stale(self, new_node: NodeInfo, stale_node_id: bytes) -> bool:
        """
        Заменить устаревший узел на новый.
        
        [KADEMLIA] Вызывается после того, как head не ответил на PING.
        
        Returns:
            True если замена произошла
        """
        with self._lock:
            if stale_node_id in self._nodes:
                del self._nodes[stale_node_id]
                self._nodes[new_node.node_id] = new_node
                self.last_updated = time.time()
                return True
            return False
    
    def needs_refresh(self, interval: float = BUCKET_REFRESH_INTERVAL) -> bool:
        """Проверить, нужно ли обновить bucket."""
        return time.time() - self.last_updated > interval


class RoutingTable:
    """
    Kademlia Routing Table - таблица маршрутизации.
    
    [KADEMLIA] Структура:
    - 160 k-buckets (по одному на каждый бит расстояния)
    - Bucket i содержит узлы с расстоянием 2^i <= d < 2^(i+1)
    - Используется для поиска k ближайших узлов к любому ID
    
    [LOOKUP] Поиск ближайших узлов:
    1. Начинаем с bucket, соответствующего целевому ID
    2. Если недостаточно узлов - расширяем на соседние buckets
    3. Сортируем по XOR-расстоянию до цели
    """
    
    def __init__(self, local_id: bytes, k: int = K):
        """
        Args:
            local_id: Наш node_id (20 байт)
            k: Размер k-bucket
        """
        if len(local_id) != 20:
            raise ValueError(f"local_id must be 20 bytes, got {len(local_id)}")
        
        self.local_id = local_id
        self.k = k
        self.buckets: List[KBucket] = [KBucket(k) for _ in range(ID_BITS)]
        self._lock = threading.Lock()
        
        logger.info(f"[DHT] RoutingTable initialized: local_id={local_id.hex()[:16]}...")
    
    def __len__(self) -> int:
        """Общее количество узлов в таблице."""
        return sum(len(b) for b in self.buckets)
    
    def get_bucket_index(self, node_id: bytes) -> int:
        """Определить bucket для узла."""
        return node_id_to_bucket_index(self.local_id, node_id)
    
    def get_bucket(self, node_id: bytes) -> KBucket:
        """Получить bucket для узла."""
        index = self.get_bucket_index(node_id)
        return self.buckets[index]
    
    def add_node(self, node: NodeInfo) -> Tuple[bool, Optional[NodeInfo]]:
        """
        Добавить узел в таблицу маршрутизации.
        
        [KADEMLIA] Логика:
        1. Определяем bucket по XOR-расстоянию
        2. Пытаемся добавить в bucket
        3. Если bucket полон - возвращаем кандидата на проверку
        
        Returns:
            (added, eviction_candidate)
        """
        # Не добавляем себя
        if node.node_id == self.local_id:
            return False, None
        
        bucket = self.get_bucket(node.node_id)
        return bucket.add(node)
    
    def remove_node(self, node_id: bytes) -> bool:
        """Удалить узел из таблицы."""
        if node_id == self.local_id:
            return False
        
        bucket = self.get_bucket(node_id)
        return bucket.remove(node_id)
    
    def get_node(self, node_id: bytes) -> Optional[NodeInfo]:
        """Получить узел по ID."""
        bucket = self.get_bucket(node_id)
        return bucket.get(node_id)
    
    def touch_node(self, node_id: bytes) -> bool:
        """Обновить время контакта узла."""
        if node_id == self.local_id:
            return False
        
        bucket = self.get_bucket(node_id)
        return bucket.touch(node_id)
    
    def find_closest(self, target_id: bytes, count: int = K, exclude: Optional[bytes] = None) -> List[NodeInfo]:
        """
        Найти k ближайших узлов к целевому ID.
        
        [KADEMLIA] Алгоритм:
        1. Собираем узлы из всех buckets
        2. Вычисляем XOR-расстояние до target_id
        3. Сортируем по расстоянию
        4. Возвращаем первые count узлов
        
        Args:
            target_id: Целевой идентификатор
            count: Количество узлов для возврата
            exclude: ID узла для исключения из результата
        
        Returns:
            Список ближайших узлов, отсортированный по расстоянию
        """
        all_nodes: List[Tuple[int, NodeInfo]] = []
        
        for bucket in self.buckets:
            for node in bucket:
                if exclude and node.node_id == exclude:
                    continue
                distance = xor_distance(target_id, node.node_id)
                all_nodes.append((distance, node))
        
        # Сортируем по расстоянию
        all_nodes.sort(key=lambda x: x[0])
        
        # Возвращаем первые count
        return [node for _, node in all_nodes[:count]]
    
    def get_refresh_ids(self) -> List[bytes]:
        """
        Получить список ID для обновления устаревших buckets.
        
        [KADEMLIA] Bucket refresh:
        - Для каждого bucket, который не обновлялся давно
        - Генерируем случайный ID в диапазоне bucket
        - Выполняем FIND_NODE для этого ID
        
        Returns:
            Список ID для lookup
        """
        refresh_ids = []
        
        for i, bucket in enumerate(self.buckets):
            if bucket.needs_refresh() and len(bucket) > 0:
                random_id = generate_random_id_in_bucket(self.local_id, i)
                refresh_ids.append(random_id)
        
        return refresh_ids
    
    def get_all_nodes(self) -> List[NodeInfo]:
        """Получить все узлы из таблицы."""
        nodes = []
        for bucket in self.buckets:
            nodes.extend(bucket.nodes)
        return nodes
    
    def get_stale_nodes(self, timeout: float = 900) -> List[NodeInfo]:
        """Получить устаревшие узлы для проверки."""
        stale = []
        for bucket in self.buckets:
            for node in bucket:
                if node.is_stale(timeout):
                    stale.append(node)
        return stale
    
    def get_stats(self) -> Dict:
        """Получить статистику таблицы маршрутизации."""
        non_empty_buckets = sum(1 for b in self.buckets if len(b) > 0)
        total_nodes = len(self)
        
        bucket_sizes = [len(b) for b in self.buckets if len(b) > 0]
        
        return {
            "local_id": self.local_id.hex(),
            "total_nodes": total_nodes,
            "non_empty_buckets": non_empty_buckets,
            "total_buckets": ID_BITS,
            "k": self.k,
            "bucket_sizes": bucket_sizes[:10],  # Первые 10 непустых
        }
    
    def to_dict(self) -> Dict:
        """Сериализация для сохранения."""
        nodes = []
        for node in self.get_all_nodes():
            nodes.append(node.to_dict())
        
        return {
            "local_id": self.local_id.hex(),
            "nodes": nodes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RoutingTable":
        """Десериализация."""
        local_id = bytes.fromhex(data["local_id"])
        table = cls(local_id)
        
        for node_data in data.get("nodes", []):
            node = NodeInfo.from_dict(node_data)
            table.add_node(node)
        
        return table


def hash_to_node_id(data: bytes) -> bytes:
    """
    Хэшировать данные в 160-битный node_id.
    
    Args:
        data: Данные для хэширования
    
    Returns:
        20-байтный SHA-1 хэш
    """
    return hashlib.sha1(data).digest()


def key_to_id(key: str) -> bytes:
    """
    Преобразовать строковый ключ в 160-битный ID для DHT.
    
    Args:
        key: Строковый ключ
    
    Returns:
        20-байтный SHA-1 хэш ключа
    """
    return hashlib.sha1(key.encode('utf-8')).digest()

