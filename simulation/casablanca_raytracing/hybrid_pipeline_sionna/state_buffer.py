from __future__ import annotations


class StateBuffer:
    def __init__(self) -> None:
        self.positions: dict[str, dict[str, float]] = {}
        self.metrics: dict[str, object] = {}
        self.handover_state: dict[str, object] = {}
        self.cell_loads: dict[int, float] = {}
        self.drone_commands: dict[int, tuple[float, float]] = {}

    def publish_positions(self, positions: dict[str, dict[str, float]]) -> None:
        self.positions = positions

    def publish_metrics(self, metrics: dict[str, object], cell_loads: dict[int, float]) -> None:
        self.metrics = metrics
        self.cell_loads = cell_loads

    def publish_handover_state(self, decisions: dict[str, object]) -> None:
        self.handover_state = decisions

    def publish_drone_commands(self, commands: dict[int, tuple[float, float]]) -> None:
        self.drone_commands = commands
