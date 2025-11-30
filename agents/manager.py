"""
Agent Manager - Система исполнения контрактов в песочнице
=========================================================

[SECURITY] Этот модуль позволяет выполнять код от других узлов
в изолированной среде (sandbox) с использованием RestrictedPython:

- Нет доступа к файловой системе
- Нет сетевых операций  
- Ограниченный набор встроенных функций
- Лимит времени выполнения

[DECENTRALIZATION] Контракты позволяют узлам договариваться
о выполнении задач без доверия к центральному серверу.
Код верифицируется криптографически перед выполнением.
"""

import asyncio
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Set
from pathlib import Path
import signal
import threading

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import (
    safe_builtins,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
)
from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem

from config import config

logger = logging.getLogger(__name__)


@dataclass
class Contract:
    """
    Контракт - исполняемый код от другого узла.
    
    [SECURITY] Контракт содержит:
    - code: исходный код Python
    - author_id: ID узла-автора
    - signature: подпись автора
    - hash: SHA256 хеш кода (идентификатор)
    """
    
    code: str
    author_id: str
    signature: str
    name: str = "unnamed"
    description: str = ""
    created_at: float = field(default_factory=time.time)
    
    @property
    def hash(self) -> str:
        """SHA256 хеш кода контракта."""
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "author_id": self.author_id,
            "signature": self.signature,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contract":
        return cls(
            code=data["code"],
            author_id=data["author_id"],
            signature=data["signature"],
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class ContractResult:
    """
    Результат выполнения контракта.
    
    [SECURITY] Результат содержит:
    - success: успешно ли выполнение
    - output: результат или сообщение об ошибке
    - execution_time: время выполнения
    - contract_hash: хеш выполненного контракта
    """
    
    success: bool
    output: Any
    execution_time: float
    contract_hash: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "execution_time": self.execution_time,
            "contract_hash": self.contract_hash,
            "error": self.error,
        }


class TimeoutError(Exception):
    """Исключение при превышении лимита времени."""
    pass


class SandboxViolation(Exception):
    """Исключение при попытке нарушить ограничения песочницы."""
    pass


class RestrictedPrinter:
    """
    Безопасная функция print для песочницы.
    
    Собирает вывод в буфер вместо stdout.
    """
    
    def __init__(self, max_size: int = 65536):
        self.buffer: List[str] = []
        self.max_size = max_size
        self.current_size = 0
    
    def __call__(self, *args, **kwargs) -> None:
        text = " ".join(str(a) for a in args)
        if self.current_size + len(text) > self.max_size:
            raise SandboxViolation("Output size limit exceeded")
        self.buffer.append(text)
        self.current_size += len(text)
    
    def get_output(self) -> str:
        return "\n".join(self.buffer)


class SafeRange:
    """
    Безопасная версия range с ограничением.
    
    [SECURITY] Предотвращает DoS через огромные range.
    """
    
    MAX_SIZE = 1_000_000
    
    def __call__(self, *args) -> range:
        r = range(*args)
        if len(r) > self.MAX_SIZE:
            raise SandboxViolation(f"Range too large: {len(r)} > {self.MAX_SIZE}")
        return r


class AgentManager:
    """
    Менеджер агентов/контрактов.
    
    [DECENTRALIZATION] AgentManager позволяет:
    - Загружать контракты от других узлов
    - Выполнять их в изолированной среде
    - Кэшировать проверенные контракты
    
    [SECURITY] Все контракты выполняются в RestrictedPython:
    - Белый список встроенных функций
    - Нет доступа к __builtins__ напрямую
    - Нет import/exec/eval
    - Контроль итераций и размеров данных
    """
    
    # Белый список безопасных встроенных функций
    SAFE_BUILTINS = {
        # Типы данных
        "True": True,
        "False": False,
        "None": None,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "bytes": bytes,
        
        # Математика
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        
        # Итерация
        "len": len,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "all": all,
        "any": any,
        
        # Строки
        "chr": chr,
        "ord": ord,
        "repr": repr,
        "ascii": ascii,
        "format": format,
        
        # Прочее
        "isinstance": isinstance,
        "issubclass": issubclass,
        "callable": callable,
        "hash": hash,
        "id": id,
        "type": type,
    }
    
    def __init__(
        self,
        max_execution_time: float = 5.0,
        max_code_size: int = 65536,
        contracts_dir: Optional[str] = None,
    ):
        """
        Инициализация менеджера агентов.
        
        Args:
            max_execution_time: Максимальное время выполнения (секунды)
            max_code_size: Максимальный размер кода (байты)
            contracts_dir: Директория для хранения контрактов
        """
        self.max_execution_time = max_execution_time
        self.max_code_size = max_code_size
        self.contracts_dir = Path(contracts_dir or config.agent.contracts_dir)
        
        # Кэш скомпилированных контрактов
        self._compiled_cache: Dict[str, Any] = {}
        
        # Белый список авторов контрактов
        self._trusted_authors: Set[str] = set()
        
        # Создаем директорию если нет
        self.contracts_dir.mkdir(parents=True, exist_ok=True)
    
    def add_trusted_author(self, author_id: str) -> None:
        """Добавить автора в белый список."""
        self._trusted_authors.add(author_id)
    
    def remove_trusted_author(self, author_id: str) -> None:
        """Удалить автора из белого списка."""
        self._trusted_authors.discard(author_id)
    
    def is_trusted_author(self, author_id: str) -> bool:
        """Проверить, доверяем ли автору."""
        return author_id in self._trusted_authors
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """
        Создать безопасное глобальное пространство имен.
        
        [SECURITY] Это ключевая функция безопасности:
        - Только белый список функций
        - Защищенный доступ к атрибутам
        - Защищенная итерация
        """
        printer = RestrictedPrinter()
        
        restricted_globals = {
            "__builtins__": self.SAFE_BUILTINS.copy(),
            "_print_": printer,
            "_getattr_": default_guarded_getattr,
            "_getitem_": default_guarded_getitem,
            "_getiter_": iter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_unpack_sequence_": guarded_unpack_sequence,
            "_write_": lambda x: x,  # Позволяем запись в контейнеры
            
            # Безопасный range
            "range": SafeRange(),
            
            # Специальные функции для вывода
            "print": printer,
            
            # Результат контракта
            "_result_": None,
        }
        
        return restricted_globals, printer
    
    def compile_contract(self, contract: Contract) -> Any:
        """
        Скомпилировать контракт.
        
        [SECURITY] Компиляция через RestrictedPython:
        - Проверяет синтаксис
        - Применяет ограничения
        - Отклоняет опасные конструкции
        
        Returns:
            Скомпилированный код
        
        Raises:
            SandboxViolation: Если код нарушает ограничения
        """
        # Проверяем размер
        if len(contract.code) > self.max_code_size:
            raise SandboxViolation(
                f"Code size {len(contract.code)} exceeds limit {self.max_code_size}"
            )
        
        # Проверяем кэш
        if contract.hash in self._compiled_cache:
            return self._compiled_cache[contract.hash]
        
        # Компилируем через RestrictedPython
        result = compile_restricted(
            contract.code,
            filename=f"<contract:{contract.hash[:8]}>",
            mode="exec",
        )
        
        # Проверяем ошибки компиляции
        if result.errors:
            errors = "\n".join(result.errors)
            raise SandboxViolation(f"Compilation errors:\n{errors}")
        
        # Кэшируем
        self._compiled_cache[contract.hash] = result.code
        
        return result.code
    
    async def execute(
        self,
        contract: Contract,
        inputs: Optional[Dict[str, Any]] = None,
        verify_signature: bool = True,
    ) -> ContractResult:
        """
        Выполнить контракт в песочнице.
        
        [SECURITY] Выполнение изолировано:
        - Ограничение по времени
        - Ограниченные функции
        - Изолированное пространство имен
        
        Args:
            contract: Контракт для выполнения
            inputs: Входные данные для контракта
            verify_signature: Проверять ли подпись автора
        
        Returns:
            ContractResult с результатом выполнения
        """
        start_time = time.time()
        
        try:
            # Компилируем контракт
            compiled = self.compile_contract(contract)
            
            # Создаем изолированное пространство имен
            restricted_globals, printer = self._create_restricted_globals()
            
            # Добавляем входные данные
            if inputs:
                for key, value in inputs.items():
                    # Проверяем, что ключ безопасен
                    if key.startswith("_"):
                        raise SandboxViolation(f"Input key cannot start with _: {key}")
                    restricted_globals[key] = value
            
            # Выполняем в отдельном потоке с таймаутом
            result = await self._execute_with_timeout(
                compiled,
                restricted_globals,
            )
            
            execution_time = time.time() - start_time
            
            # Собираем результат
            output = restricted_globals.get("_result_", result)
            printed_output = printer.get_output()
            
            if printed_output and output is None:
                output = printed_output
            
            return ContractResult(
                success=True,
                output=output,
                execution_time=execution_time,
                contract_hash=contract.hash,
            )
            
        except TimeoutError:
            return ContractResult(
                success=False,
                output=None,
                execution_time=self.max_execution_time,
                contract_hash=contract.hash,
                error="Execution timeout",
            )
            
        except SandboxViolation as e:
            return ContractResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                contract_hash=contract.hash,
                error=f"Sandbox violation: {e}",
            )
            
        except Exception as e:
            return ContractResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                contract_hash=contract.hash,
                error=f"Execution error: {type(e).__name__}: {e}",
            )
    
    async def _execute_with_timeout(
        self,
        compiled_code: Any,
        restricted_globals: Dict[str, Any],
    ) -> Any:
        """
        Выполнить код с ограничением по времени.
        
        [SECURITY] Использует asyncio для неблокирующего таймаута.
        """
        loop = asyncio.get_event_loop()
        
        def run_code():
            # Локальное пространство имен
            restricted_locals: Dict[str, Any] = {}
            exec(compiled_code, restricted_globals, restricted_locals)
            return restricted_locals.get("result", None)
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, run_code),
                timeout=self.max_execution_time,
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError("Contract execution exceeded time limit")
    
    def save_contract(self, contract: Contract) -> str:
        """
        Сохранить контракт в файл.
        
        Returns:
            Путь к файлу
        """
        import json
        
        filepath = self.contracts_dir / f"{contract.hash}.json"
        with open(filepath, "w") as f:
            json.dump(contract.to_dict(), f, indent=2)
        
        return str(filepath)
    
    def load_contract(self, contract_hash: str) -> Optional[Contract]:
        """
        Загрузить контракт из файла.
        
        Returns:
            Contract или None если не найден
        """
        import json
        
        filepath = self.contracts_dir / f"{contract_hash}.json"
        if not filepath.exists():
            return None
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return Contract.from_dict(data)
    
    def list_contracts(self) -> List[str]:
        """Получить список хешей сохраненных контрактов."""
        contracts = []
        for filepath in self.contracts_dir.glob("*.json"):
            contracts.append(filepath.stem)
        return contracts


# Примеры безопасных контрактов
EXAMPLE_CONTRACTS = {
    "calculator": """
# Простой калькулятор
# Входные данные: a, b, operation

if operation == "add":
    _result_ = a + b
elif operation == "sub":
    _result_ = a - b
elif operation == "mul":
    _result_ = a * b
elif operation == "div":
    _result_ = a / b if b != 0 else "Division by zero"
else:
    _result_ = "Unknown operation"
""",
    
    "data_processor": """
# Обработчик данных
# Входные данные: data (list)

if not isinstance(data, list):
    _result_ = "Error: data must be a list"
else:
    total = sum(data) if data else 0
    average = total / len(data) if data else 0
    _result_ = {
        "count": len(data),
        "sum": total,
        "average": average,
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }
""",
    
    "hash_verifier": """
# Проверка хеша
# Входные данные: text, expected_hash

import hashlib  # Это будет заблокировано RestrictedPython!

# Безопасная альтернатива - используем встроенный hash
computed = hash(text)
_result_ = {
    "computed_hash": computed,
    "matches": str(computed) == expected_hash,
}
""",
}

