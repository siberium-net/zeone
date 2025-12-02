"""
P2P Network Configuration
=========================
Централизованная конфигурация для всех модулей сети.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class NetworkConfig:
    """Настройки сетевого слоя."""
    
    # Порт по умолчанию для TCP сервера
    default_port: int = 8468
    
    # Bootstrap узлы для первоначального подключения к сети
    # [DECENTRALIZATION] Это единственная "централизованная" точка входа.
    # После подключения узел получает список других пиров и может работать
    # независимо от bootstrap-узлов.
    bootstrap_nodes: List[Tuple[str, int]] = field(default_factory=lambda: [
        ("127.0.0.1", 8468),  # Локальный узел для тестирования
        # Добавьте публичные bootstrap-узлы здесь
    ])
    
    # UDP broadcast порт для LAN discovery
    broadcast_port: int = 8469
    
    # Таймаут подключения (секунды)
    connection_timeout: float = 10.0
    
    # Таймаут RPC запросов (секунды)
    rpc_timeout: float = 5.0
    
    # Интервал heartbeat для проверки живости пиров (секунды)
    heartbeat_interval: float = 30.0
    
    # Максимальное количество активных соединений
    max_peers: int = 50
    
    # Размер буфера для чтения данных
    buffer_size: int = 65536


@dataclass
class CryptoConfig:
    """Настройки криптографии."""
    
    # Путь к файлу с ключевой парой
    identity_file: str = "identity.key"
    
    # Алгоритм подписи: Ed25519 (через PyNaCl)
    # Алгоритм шифрования: Curve25519-XSalsa20-Poly1305 (NaCl Box)


@dataclass
class LedgerConfig:
    """Настройки экономического модуля."""
    
    # Путь к базе данных SQLite
    database_path: str = "ledger.db"
    
    # Начальный Trust Score для новых пиров
    initial_trust_score: float = 0.5
    
    # Минимальный Trust Score для взаимодействия
    min_trust_score: float = 0.1
    
    # Множитель увеличения Trust Score за успешную транзакцию
    trust_increase_factor: float = 0.01
    
    # Множитель уменьшения Trust Score за неудачу
    trust_decrease_factor: float = 0.05


@dataclass
class AgentConfig:
    """Настройки системы агентов/контрактов."""
    
    # Максимальное время выполнения контракта (секунды)
    max_execution_time: float = 5.0
    
    # Максимальный размер кода контракта (байты)
    max_code_size: int = 65536
    
    # Директория для хранения загруженных контрактов
    contracts_dir: str = "contracts"


@dataclass
class PersistenceConfig:
    """[PRODUCTION] Настройки persistence."""
    
    # Путь к базе состояния
    state_db_path: str = "node_state.db"
    
    # Интервал автосохранения (секунды)
    auto_save_interval: float = 60.0
    
    # Время жизни неактивных пиров (секунды)
    peer_stale_threshold: float = 3600.0
    
    # Максимум пиров в кэше
    max_cached_peers: int = 1000


@dataclass
class SecurityConfig:
    """[PRODUCTION] Настройки безопасности."""
    
    # Rate Limiting
    rate_limit_messages_per_sec: float = 20.0
    rate_limit_connections_per_min: int = 10
    rate_limit_burst_size: int = 100
    
    # DoS Protection
    dos_max_bandwidth_per_sec: int = 1_000_000  # 1 MB/s
    dos_max_invalid_ratio: float = 0.3
    dos_temp_ban_duration: float = 300.0  # 5 min
    dos_perm_ban_threshold: int = 5
    
    # Whitelist (node_ids)
    whitelist: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """[PRODUCTION] Настройки мониторинга."""
    
    # Health Checks
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    unhealthy_threshold: int = 3
    
    # Metrics
    metrics_prefix: str = "p2p"
    metrics_export_interval: float = 60.0
    
    # HTTP endpoint для health/metrics
    http_port: int = 8080
    http_enabled: bool = False


@dataclass
class Config:
    """Главный конфигурационный класс."""
    
    network: NetworkConfig = field(default_factory=NetworkConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


# Глобальный экземпляр конфигурации
config = Config()
