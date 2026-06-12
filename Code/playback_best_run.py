"""
Play back a Linesight .inputs file through a running TMNF+TMInterface instance
and wait for the replay to be saved as a personal best (.Replay.Gbx).

Usage (from ProjectApex-Linesight/):
    python playback_best_run.py <path_to_inputs_file>

Example:
    python playback_best_run.py linesight/save/ovaltrack1_run01/best_runs/oval1_30900/oval1_30900.inputs
"""

import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent / "linesight"
sys.path.insert(0, str(REPO))

from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface
from trackmania_rl.map_loader import map_name_from_map_path
from config_files import user_config as u

PORT = u.base_tmi_port  # 8478


def launch_tmnf():
    """Launch TMNF via TMLoader with the configured profile (same as training does)."""
    if u.is_linux:
        subprocess.Popen([str(u.linux_launch_game_path), str(PORT)])
        return
    launch_string = (
        'powershell -executionPolicy bypass -command "& {'
        f" start-process -FilePath '{u.windows_TMLoader_path}'"
        " -ArgumentList "
        f'\'run TmForever "{u.windows_TMLoader_profile_name}" /configstring=\\"set custom_port {PORT}\\"\''
        '}"'
    )
    subprocess.Popen(launch_string)


# Map TMNF challenge paths by the short name Linesight uses in best_runs/ folders.
# Folder names look like "monza_100610", "fig8_43560", "oval1_30900", everything
# before the underscore identifies the map.
MAP_PATH_BY_SHORT = {
    "monza":      '"My Challenges/Monza.Challenge.Gbx"',
    "fig8":       '"My Challenges/Figure8Track.Challenge.Gbx"',
    "map5":       '"My Challenges/Map5.Challenge.Gbx"',
    "oval1":      '"My Challenges/ovaltrack1.Challenge.Gbx"',
    "ovaltrack1": '"My Challenges/ovaltrack1.Challenge.Gbx"',
    "hock":       '"ESL-Hockolicious.Challenge.Gbx"',
}


def _all_map_paths() -> dict[str, str]:
    """Return MAP_PATH_BY_SHORT merged with any maps registered in maps_registry.json."""
    result = dict(MAP_PATH_BY_SHORT)
    registry = Path(__file__).resolve().parent / "maps_registry.json"
    if registry.exists():
        try:
            data = json.loads(registry.read_text(encoding="utf-8"))
            for m in data.get("maps", []):
                short = m.get("short", "")
                path = m.get("challenge_path", "")
                if short and path and short not in result:
                    result[short] = f'"{path}"'
        except Exception:
            pass
    return result


def map_path_for_inputs(run_name: str) -> str:
    """Derive the TMNF map path from a best_runs folder name like 'monza_100610'."""
    short = run_name.split("_", 1)[0]
    all_paths = _all_map_paths()
    if short not in all_paths:
        raise SystemExit(
            f"Unknown map short name '{short}' (from run '{run_name}'). "
            f"Register the map in the dashboard's Hyperparameters -> Map Registry section."
        )
    return all_paths[short]


def hide_pb_replay(map_path_quoted: str) -> Path | None:
    """If TMNF has a personal-best autosave for this map, rename it to .bak so
    TMNF starts the map in a clean 'race attempt' state. Without this, TMNF
    loads the map in 'challenge with PB ghost' mode and the script's race-start
    handshake times out.

    Linesight's own training loop does this via map_loader.hide_personal_record_replay,
    but that helper uses os.getlogin() (Windows username) instead of the TMNF
    in-game username. On a typical setup the two differ, we use the in-game
    username from user_config so the file is actually found and hidden.
    """
    try:
        inner = map_name_from_map_path(map_path_quoted)
    except Exception as e:
        print(f"  [hide_pb] couldn't read inner map name from {map_path_quoted}: {e}")
        return None
    autosaves = u.trackmania_base_path / "Tracks" / "Replays" / "Autosaves"
    pb_file = autosaves / f"{u.username}_{inner}.Replay.gbx"
    if not pb_file.exists():
        print(f"  [hide_pb] no PB autosave to hide ({pb_file.name} doesn't exist).")
        return None
    bak = pb_file.with_suffix(pb_file.suffix + ".bak")
    pb_file.replace(bak)
    print(f"  [hide_pb] hid {pb_file.name} -> {bak.name}")
    return bak


def main():
    if len(sys.argv) < 2:
        print("Usage: python playback_best_run.py <path_to_inputs_file>")
        sys.exit(1)

    inputs_path = Path(sys.argv[1]).resolve()
    if not inputs_path.exists():
        print(f"ERROR: inputs file not found: {inputs_path}")
        sys.exit(1)

    inputs_folder = str(inputs_path.parent)
    inputs_filename = inputs_path.name
    run_name = inputs_path.stem  # e.g. "oval1_30900"

    replays_dir = u.trackmania_base_path / "Tracks" / "Replays"
    out_dir = inputs_path.parent.parent / (inputs_path.parent.name + "_replays")
    out_dir.mkdir(exist_ok=True)

    print(f"Connecting to TMInterface on port {PORT} ...")
    iface = TMInterface(PORT)

    deadline = time.monotonic() + 60
    while not iface.registered:
        try:
            iface.register(10)
            break
        except ConnectionRefusedError as e:
            if time.monotonic() > deadline:
                print("FAIL: could not connect after 60s. Check TMLoader path in Settings.")
                sys.exit(1)
            time.sleep(0.5)

    print("Connected.")

    # Pre-compute map path so we can re-send the map command if needed
    map_path_quoted = map_path_for_inputs(run_name)
    map_cmd = f"map {map_path_quoted}"
    hide_pb_replay(map_path_quoted)

    map_loaded = False
    inputs_loaded = False
    run_finished = False
    connect_count = 0

    while not run_finished:
        msgtype = iface._read_int32()

        if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
            connect_count += 1
            if not inputs_loaded:
                # Map command not yet acted on, send/resend it.
                # This handles both the first connect and the case where the
                # first connect fired while the game was still on the profile
                # selection screen (command ignored), so we retry on the next
                # SC_ON_CONNECT_SYNC that arrives once the game is at the menu.
                print(f"Connect #{connect_count}: loading map ...")
                iface.execute_command("set speed 1")
                iface.execute_command(f'set autologin {u.username}')
                iface.execute_command(f'set scripts_folder {inputs_folder}')
                print(f"  -> {map_cmd}")
                iface.execute_command(map_cmd)
                map_loaded = True
                print('Map load requested, waiting for race to start ...')
            else:
                # Race already started, this is a mid-race reconnect after map load
                print(f"Reconnected (connect #{connect_count}), race is resetting ...")
            iface._respond_to_call(msgtype)

        elif msgtype == int(MessageType.SC_RUN_STEP_SYNC):
            t = iface._read_int32()
            if map_loaded and not inputs_loaded:
                print(f"Race running (t={t}ms), loading inputs: {inputs_filename} ...")
                iface.execute_command(f"load {inputs_filename}")
                inputs_loaded = True
            iface._respond_to_call(msgtype)

        elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
            current = iface._read_int32()
            target = iface._read_int32()
            print(f"  Checkpoint {current}/{target}")
            if current == target:
                print("Run finished! Waiting for replay file ...")
                iface.execute_command("finish")
                iface.close()
                run_finished = True
            else:
                iface._respond_to_call(msgtype)

        elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
            lap = iface._read_int32()
            iface._read_int32()
            print(f"  Lap {lap}")
            iface._respond_to_call(msgtype)

        elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
            iface._respond_to_call(msgtype)

        elif msgtype == int(MessageType.C_SHUTDOWN):
            iface.close()
            break

    # Wait for TMNF to write the personal-best replay
    print("Waiting for .Replay.Gbx to appear ...")
    deadline = time.monotonic() + 30
    found = None
    while time.monotonic() < deadline:
        candidates = sorted(replays_dir.glob("*.Replay.Gbx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates and (time.monotonic() - candidates[0].stat().st_mtime) < 15:
            found = candidates[0]
            break
        time.sleep(0.5)

    if found:
        dest = out_dir / f"{run_name}.Replay.Gbx"
        shutil.copy2(found, dest)
        print(f"\nSaved replay to: {dest}")
        print("You can now load this in TMNF via: Replays menu -> browse to the file")
    else:
        print("\nReplay file not found automatically.")
        print(f"Check {replays_dir} manually, TMNF may have saved it there.")


if __name__ == "__main__":
    main()
