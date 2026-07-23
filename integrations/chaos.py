from __future__ import annotations

import ipaddress
import os
import re
import subprocess
import threading
from collections.abc import Callable


class OutputPortReject:
    """A narrowly scoped kernel-level connection partition with idempotent cleanup."""

    def __init__(
        self,
        host: str,
        port: int,
        label: str,
        *,
        uid: int | None = None,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        ipaddress.ip_address(host)
        if not 1 <= int(port) <= 65535:
            raise ValueError("port is out of range")
        if re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", label) is None:
            raise ValueError("invalid partition label")
        self.host = host
        self.port = int(port)
        self.label = label
        self.uid = os.getuid() if uid is None else int(uid)
        self._runner = runner
        self._installed = False
        self._lock = threading.Lock()

    def _rule(self, operation: str) -> list[str]:
        command = (
            ["iptables", operation, "OUTPUT"]
            if os.geteuid() == 0
            else ["sudo", "-n", "iptables", operation, "OUTPUT"]
        )
        if operation == "-I":
            command.append("1")
        command.extend(
            [
                "-p",
                "tcp",
                "-d",
                self.host,
                "--dport",
                str(self.port),
                "-m",
                "owner",
                "--uid-owner",
                str(self.uid),
                "-m",
                "comment",
                "--comment",
                self.label,
                "-j",
                "REJECT",
                "--reject-with",
                "tcp-reset",
            ]
        )
        return command

    def install(self) -> None:
        with self._lock:
            if self._installed:
                return
            self._runner(self._rule("-I"), check=True, capture_output=True, text=True)
            self._installed = True

    def remove(self) -> None:
        with self._lock:
            if not self._installed:
                return
            self._runner(self._rule("-D"), check=True, capture_output=True, text=True)
            self._installed = False


def schedule_partition_removal(
    partition: OutputPortReject,
    seconds: float,
    on_removed: Callable[[], None],
) -> threading.Timer:
    if seconds <= 0:
        raise ValueError("partition duration must be positive")

    def remove() -> None:
        partition.remove()
        on_removed()

    timer = threading.Timer(seconds, remove)
    timer.daemon = True
    timer.start()
    return timer
