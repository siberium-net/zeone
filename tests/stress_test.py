#!/usr/bin/env python3
"""
ZEONE Stress Test Suite
=======================

[STRESS] Load testing for P2P node under pressure.

Scenarios:
1. Protocol Flood: Bombard with PING/PONG messages
2. DHT Storm: Massive FIND_NODE / FIND_VALUE requests
3. Cache Assault: Concurrent CACHE_REQUEST operations
4. Connection Churn: Rapid connect/disconnect cycles

Goals:
- Find memory leaks in asyncio handlers
- Detect race conditions in concurrent access
- Measure throughput limits
- Identify breaking points

Usage:
    # Run stress tests (not included in pytest by default)
    python tests/stress_test.py --clients 100 --duration 60
    
    # Or via pytest (marked slow)
    pytest tests/stress_test.py -v -m stress
"""

import asyncio
import argparse
import logging
import time
import sys
import os
import random
import tracemalloc
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import asynccontextmanager

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StressConfig:
    """Stress test configuration."""
    target_host: str = "127.0.0.1"
    target_port: int = 8468
    num_clients: int = 100
    duration_seconds: float = 60.0
    ramp_up_seconds: float = 10.0
    cooldown_seconds: float = 5.0
    message_rate_per_client: float = 10.0  # msgs/sec
    connection_timeout: float = 5.0
    request_timeout: float = 2.0
    enable_memory_tracking: bool = True
    log_interval_seconds: float = 5.0
    scenarios: List[str] = field(default_factory=lambda: ["ping", "dht", "cache"])


@dataclass
class StressMetrics:
    """Collected stress test metrics."""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_errors: int = 0
    
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    latencies_ms: List[float] = field(default_factory=list)
    
    start_time: float = 0.0
    end_time: float = 0.0
    
    peak_memory_mb: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def messages_per_second(self) -> float:
        if self.duration > 0:
            return self.total_messages_sent / self.duration
        return 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        if self.latencies_ms:
            return sum(self.latencies_ms) / len(self.latencies_ms)
        return 0.0
    
    @property
    def p99_latency_ms(self) -> float:
        if self.latencies_ms:
            sorted_latencies = sorted(self.latencies_ms)
            idx = int(len(sorted_latencies) * 0.99)
            return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(self.duration, 2),
            "connections": {
                "total": self.total_connections,
                "successful": self.successful_connections,
                "failed": self.failed_connections,
            },
            "messages": {
                "sent": self.total_messages_sent,
                "received": self.total_messages_received,
                "errors": self.total_errors,
                "rate_per_second": round(self.messages_per_second, 2),
            },
            "traffic": {
                "bytes_sent": self.total_bytes_sent,
                "bytes_received": self.total_bytes_received,
            },
            "latency_ms": {
                "avg": round(self.avg_latency_ms, 2),
                "p99": round(self.p99_latency_ms, 2),
            },
            "memory_mb": {
                "peak": round(self.peak_memory_mb, 2),
            },
            "errors_by_type": dict(self.errors_by_type),
        }


# ============================================================================
# Lightweight Stress Client
# ============================================================================

class StressClient:
    """
    Lightweight stress test client.
    
    [MINIMAL] Uses raw sockets with minimal protocol overhead.
    Does not use full Node stack to maximize load generation.
    """
    
    def __init__(self, client_id: int, config: StressConfig, metrics: StressMetrics):
        self.client_id = client_id
        self.config = config
        self.metrics = metrics
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        self.running = False
        self._message_counter = 0
        
        # Random node ID for this client
        self.node_id = os.urandom(32).hex()
    
    async def connect(self) -> bool:
        """Connect to target node."""
        self.metrics.total_connections += 1
        
        try:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.config.target_host,
                    self.config.target_port,
                ),
                timeout=self.config.connection_timeout,
            )
            
            # Set TCP_NODELAY for lower latency
            sock = self.writer.get_extra_info("socket")
            if sock:
                import socket
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            self.connected = True
            self.metrics.successful_connections += 1
            return True
            
        except asyncio.TimeoutError:
            self.metrics.failed_connections += 1
            self.metrics.errors_by_type["connection_timeout"] += 1
            return False
        except Exception as e:
            self.metrics.failed_connections += 1
            self.metrics.errors_by_type[type(e).__name__] += 1
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from target."""
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
        
        self.connected = False
        self.reader = None
        self.writer = None
    
    def _build_message(self, msg_type: str) -> bytes:
        """Build a test message."""
        self._message_counter += 1
        
        # Try to use real binary protocol
        try:
            from core.wire import WireMessageType, BinaryWireProtocol
            from core.transport import Crypto
            
            # Create minimal crypto for signing
            crypto = Crypto()
            
            if msg_type == "ping":
                mtype = WireMessageType.PING
                payload = b""
            elif msg_type == "find_node":
                mtype = WireMessageType.FIND_NODE
                # Random target ID
                payload = os.urandom(20)
            elif msg_type == "cache_request":
                mtype = WireMessageType.CACHE_REQUEST
                # Random chunk hash
                payload = os.urandom(32)
            else:
                mtype = WireMessageType.DATA
                payload = f"stress_{self._message_counter}".encode()
            
            return BinaryWireProtocol.pack_message(
                msg_type=mtype,
                payload=payload,
                signing_key=crypto._signing_key,
            )
            
        except ImportError:
            # Fallback: Raw binary message
            header = b"ZE" + bytes([1, 1, 0, 0, 0, 8])  # Magic, Ver, Type, Flags, Len
            payload = f"test_{self._message_counter}".encode()[:8].ljust(8, b"\x00")
            return header + bytes(24) + bytes(64) + payload  # + nonce + sig
    
    async def send_message(self, msg_type: str = "ping") -> Optional[float]:
        """
        Send a message and measure round-trip time.
        
        Returns:
            Latency in milliseconds, or None if failed
        """
        if not self.connected or not self.writer:
            return None
        
        message = self._build_message(msg_type)
        start = time.perf_counter()
        
        try:
            # Send
            self.writer.write(message)
            await asyncio.wait_for(
                self.writer.drain(),
                timeout=self.config.request_timeout,
            )
            
            self.metrics.total_messages_sent += 1
            self.metrics.total_bytes_sent += len(message)
            
            # Try to receive response
            if self.reader:
                try:
                    response = await asyncio.wait_for(
                        self.reader.read(1024),
                        timeout=self.config.request_timeout,
                    )
                    
                    if response:
                        self.metrics.total_messages_received += 1
                        self.metrics.total_bytes_received += len(response)
                        
                        latency = (time.perf_counter() - start) * 1000
                        return latency
                    
                except asyncio.TimeoutError:
                    # No response within timeout - could be fire-and-forget
                    pass
            
            return None
            
        except asyncio.TimeoutError:
            self.metrics.total_errors += 1
            self.metrics.errors_by_type["send_timeout"] += 1
            return None
        except Exception as e:
            self.metrics.total_errors += 1
            self.metrics.errors_by_type[type(e).__name__] += 1
            return None
    
    async def run_scenario(self, scenario: str, duration: float) -> None:
        """Run stress scenario for specified duration."""
        self.running = True
        end_time = time.time() + duration
        interval = 1.0 / self.config.message_rate_per_client
        
        msg_types = {
            "ping": ["ping"],
            "dht": ["find_node"],
            "cache": ["cache_request"],
            "mixed": ["ping", "find_node", "cache_request"],
        }
        
        types_to_send = msg_types.get(scenario, ["ping"])
        
        while self.running and time.time() < end_time:
            msg_type = random.choice(types_to_send)
            latency = await self.send_message(msg_type)
            
            if latency is not None:
                self.metrics.latencies_ms.append(latency)
            
            # Rate limiting
            await asyncio.sleep(interval)
    
    async def stop(self) -> None:
        """Stop client."""
        self.running = False
        await self.disconnect()


# ============================================================================
# Stress Test Orchestrator
# ============================================================================

class StressTestOrchestrator:
    """
    Orchestrates stress test execution.
    
    [PHASES]
    1. Ramp-up: Gradually spin up clients
    2. Steady-state: Full load
    3. Cooldown: Graceful shutdown
    """
    
    def __init__(self, config: StressConfig):
        self.config = config
        self.metrics = StressMetrics()
        self.clients: List[StressClient] = []
        self.logger = logging.getLogger("stress")
    
    async def _memory_monitor(self) -> None:
        """Background task to monitor memory."""
        if not self.config.enable_memory_tracking:
            return
        
        while True:
            import tracemalloc
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024
            
            self.metrics.memory_samples.append(current_mb)
            self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, peak_mb)
            
            await asyncio.sleep(1.0)
    
    async def _metrics_reporter(self) -> None:
        """Background task to log metrics."""
        last_sent = 0
        
        while True:
            await asyncio.sleep(self.config.log_interval_seconds)
            
            msgs_sent = self.metrics.total_messages_sent
            rate = (msgs_sent - last_sent) / self.config.log_interval_seconds
            last_sent = msgs_sent
            
            active = sum(1 for c in self.clients if c.connected)
            
            self.logger.info(
                f"[STRESS] Active: {active}/{len(self.clients)}, "
                f"Msgs: {msgs_sent}, Rate: {rate:.1f}/s, "
                f"Errors: {self.metrics.total_errors}, "
                f"Memory: {self.metrics.peak_memory_mb:.1f}MB"
            )
    
    async def run(self) -> StressMetrics:
        """Execute stress test."""
        self.logger.info(f"[STRESS] Starting stress test")
        self.logger.info(f"[STRESS] Target: {self.config.target_host}:{self.config.target_port}")
        self.logger.info(f"[STRESS] Clients: {self.config.num_clients}")
        self.logger.info(f"[STRESS] Duration: {self.config.duration_seconds}s")
        
        # Start memory tracking
        if self.config.enable_memory_tracking:
            tracemalloc.start()
        
        self.metrics.start_time = time.time()
        
        # Start background tasks
        tasks = []
        tasks.append(asyncio.create_task(self._memory_monitor()))
        tasks.append(asyncio.create_task(self._metrics_reporter()))
        
        try:
            # Phase 1: Ramp-up
            self.logger.info(f"[STRESS] Phase 1: Ramp-up ({self.config.ramp_up_seconds}s)")
            
            ramp_interval = self.config.ramp_up_seconds / self.config.num_clients
            
            for i in range(self.config.num_clients):
                client = StressClient(i, self.config, self.metrics)
                self.clients.append(client)
                
                # Connect in background
                asyncio.create_task(self._run_client(client))
                
                await asyncio.sleep(ramp_interval)
            
            # Phase 2: Steady state
            self.logger.info(f"[STRESS] Phase 2: Steady state ({self.config.duration_seconds}s)")
            await asyncio.sleep(self.config.duration_seconds)
            
            # Phase 3: Cooldown
            self.logger.info(f"[STRESS] Phase 3: Cooldown ({self.config.cooldown_seconds}s)")
            for client in self.clients:
                await client.stop()
            
            await asyncio.sleep(self.config.cooldown_seconds)
            
        finally:
            # Cancel background tasks
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup
            for client in self.clients:
                await client.disconnect()
            
            if self.config.enable_memory_tracking:
                tracemalloc.stop()
        
        self.metrics.end_time = time.time()
        
        return self.metrics
    
    async def _run_client(self, client: StressClient) -> None:
        """Run a single client's lifecycle."""
        if not await client.connect():
            return
        
        # Pick random scenario
        scenario = random.choice(self.config.scenarios)
        
        try:
            await client.run_scenario(
                scenario,
                self.config.duration_seconds,
            )
        except Exception as e:
            self.logger.debug(f"Client {client.client_id} error: {e}")
        finally:
            await client.disconnect()


# ============================================================================
# Report Generator
# ============================================================================

def print_report(metrics: StressMetrics) -> None:
    """Print stress test report."""
    print("\n" + "=" * 60)
    print("  ZEONE STRESS TEST REPORT")
    print("=" * 60)
    
    data = metrics.to_dict()
    
    print(f"\n[DURATION]")
    print(f"  Total time: {data['duration_seconds']}s")
    
    print(f"\n[CONNECTIONS]")
    print(f"  Total:      {data['connections']['total']}")
    print(f"  Successful: {data['connections']['successful']}")
    print(f"  Failed:     {data['connections']['failed']}")
    
    print(f"\n[MESSAGES]")
    print(f"  Sent:       {data['messages']['sent']}")
    print(f"  Received:   {data['messages']['received']}")
    print(f"  Errors:     {data['messages']['errors']}")
    print(f"  Rate:       {data['messages']['rate_per_second']:.1f} msg/s")
    
    print(f"\n[TRAFFIC]")
    print(f"  Sent:       {data['traffic']['bytes_sent'] / 1024:.1f} KB")
    print(f"  Received:   {data['traffic']['bytes_received'] / 1024:.1f} KB")
    
    print(f"\n[LATENCY]")
    print(f"  Average:    {data['latency_ms']['avg']:.2f} ms")
    print(f"  P99:        {data['latency_ms']['p99']:.2f} ms")
    
    print(f"\n[MEMORY]")
    print(f"  Peak:       {data['memory_mb']['peak']:.1f} MB")
    
    if data['errors_by_type']:
        print(f"\n[ERRORS BY TYPE]")
        for err_type, count in sorted(data['errors_by_type'].items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count}")
    
    print("\n" + "=" * 60)
    
    # Pass/Fail assessment
    error_rate = data['messages']['errors'] / max(data['messages']['sent'], 1)
    
    if error_rate < 0.01:  # <1% errors
        print("  RESULT: [PASS] Error rate below 1%")
    elif error_rate < 0.05:  # <5% errors
        print("  RESULT: [WARN] Error rate below 5%")
    else:
        print("  RESULT: [FAIL] Error rate above 5%")
    
    print("=" * 60 + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main_async(config: StressConfig) -> int:
    """Async main entry point."""
    orchestrator = StressTestOrchestrator(config)
    metrics = await orchestrator.run()
    print_report(metrics)
    
    # Return exit code based on results
    error_rate = metrics.total_errors / max(metrics.total_messages_sent, 1)
    return 0 if error_rate < 0.05 else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ZEONE Stress Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic stress test
    python tests/stress_test.py
    
    # High load
    python tests/stress_test.py --clients 500 --duration 120
    
    # Target remote node
    python tests/stress_test.py --host 192.168.1.100 --port 8468
""",
    )
    
    parser.add_argument("--host", default="127.0.0.1", help="Target host")
    parser.add_argument("--port", type=int, default=8468, help="Target port")
    parser.add_argument("--clients", type=int, default=100, help="Number of clients")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration (seconds)")
    parser.add_argument("--rate", type=float, default=10.0, help="Messages per second per client")
    parser.add_argument("--ramp-up", type=float, default=10.0, help="Ramp-up time (seconds)")
    parser.add_argument("--scenarios", default="ping,dht,cache", help="Scenarios (comma-separated)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    config = StressConfig(
        target_host=args.host,
        target_port=args.port,
        num_clients=args.clients,
        duration_seconds=args.duration,
        message_rate_per_client=args.rate,
        ramp_up_seconds=args.ramp_up,
        scenarios=args.scenarios.split(","),
    )
    
    return asyncio.run(main_async(config))


if __name__ == "__main__":
    sys.exit(main())

