from __future__ import annotations

import struct

STATE_MAGIC = b"HYB1"
CONTROL_MAGIC = b"HYB2"

STATE_HEADER = struct.Struct("<4sIIII")
ENTITY_RECORD = struct.Struct("<ffffff")
CONTROL_HEADER = struct.Struct("<4sIIII")
VELOCITY_UPDATE = struct.Struct("<Ifff")
WAYPOINT_UPDATE = struct.Struct("<Iffff")


def pack_state(step: int, time_ms: int, ue_states, drone_states) -> bytes:
    payload = bytearray()
    payload += STATE_HEADER.pack(STATE_MAGIC, step, time_ms, len(ue_states), len(drone_states))
    for state in ue_states:
        payload += ENTITY_RECORD.pack(*state)
    for state in drone_states:
        payload += ENTITY_RECORD.pack(*state)
    return bytes(payload)


def unpack_state(payload: bytes):
    magic, step, time_ms, ue_count, drone_count = STATE_HEADER.unpack_from(payload, 0)
    if magic != STATE_MAGIC:
        raise ValueError(f"Unexpected state magic {magic!r}")
    offset = STATE_HEADER.size
    ue_states = []
    for _ in range(ue_count):
        ue_states.append(ENTITY_RECORD.unpack_from(payload, offset))
        offset += ENTITY_RECORD.size
    drone_states = []
    for _ in range(drone_count):
        drone_states.append(ENTITY_RECORD.unpack_from(payload, offset))
        offset += ENTITY_RECORD.size
    return step, time_ms, ue_states, drone_states


def pack_control(step: int, stop_flag: int, ue_velocity_updates, drone_waypoints) -> bytes:
    payload = bytearray()
    payload += CONTROL_HEADER.pack(CONTROL_MAGIC, step, stop_flag, len(ue_velocity_updates), len(drone_waypoints))
    for update in ue_velocity_updates:
        payload += VELOCITY_UPDATE.pack(*update)
    for waypoint in drone_waypoints:
        payload += WAYPOINT_UPDATE.pack(*waypoint)
    return bytes(payload)


def unpack_control(payload: bytes):
    magic, step, stop_flag, ue_updates_count, drone_waypoint_count = CONTROL_HEADER.unpack_from(payload, 0)
    if magic != CONTROL_MAGIC:
        raise ValueError(f"Unexpected control magic {magic!r}")
    offset = CONTROL_HEADER.size
    ue_updates = []
    for _ in range(ue_updates_count):
        ue_updates.append(VELOCITY_UPDATE.unpack_from(payload, offset))
        offset += VELOCITY_UPDATE.size
    drone_waypoints = []
    for _ in range(drone_waypoint_count):
        drone_waypoints.append(WAYPOINT_UPDATE.unpack_from(payload, offset))
        offset += WAYPOINT_UPDATE.size
    return step, stop_flag, ue_updates, drone_waypoints
