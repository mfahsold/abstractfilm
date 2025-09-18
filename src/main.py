from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.configuration import build_config, load_preset
from src.pipeline import pipeline_from_config


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Abstract Morphing Artfilm Pipeline")
    parser.add_argument("--input", help="Pfad zum Eingabevideo")
    parser.add_argument("--output", help="Pfad zur Ausgabedatei (MP4)")
    parser.add_argument("--preset", help="Preset-Datei (JSON/YAML)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--preview", action="store_true", help="Aktiviere Preview-Voreinstellungen")
    mode_group.add_argument("--full", action="store_true", help="Aktiviere Produktions-Voreinstellungen")

    parser.add_argument("--scene-detect", action="store_true", help="Shot-Erkennung aktivieren")
    parser.add_argument("--include-audio", action="store_true", help="Originalaudio übernehmen")
    parser.add_argument("--start", type=float, help="Startzeit in Sekunden")
    parser.add_argument("--duration", type=float, help="Dauer in Sekunden")
    parser.add_argument("--width", type=int, help="Ausgabe-Breite")
    parser.add_argument("--height", type=int, help="Ausgabe-Höhe")
    parser.add_argument("--flow-algo", choices=["DIS", "FARNEBACK"], help="Optical-Flow-Algorithmus")
    parser.add_argument("--flow-preset", choices=["ULTRAFAST", "FAST", "MEDIUM"], help="DIS-Preset")
    parser.add_argument("--rd-steps", type=int, help="Reaction-Diffusion-Schritte pro Frame")
    parser.add_argument("--glow", action=argparse.BooleanOptionalAction, default=None, help="Glow einschalten")
    parser.add_argument("--glow-strength", type=float, help="Glow-Verstärkung")
    parser.add_argument("--use-lut", help="Pfad zu 3D-LUT (.cube)")
    parser.add_argument("--morph-alpha", type=float, help="Blendfaktor zwischen Frames")
    parser.add_argument("--flow-scale", type=float, help="Skalierung des Flussfeldes")
    parser.add_argument("--random-seed", type=int, help="Seed für reproduzierbare Effekte")
    parser.add_argument("--preview-mode", action="store_true", help="Preview optimieren")
    parser.add_argument("--effects-profile", help="Pfad zu Effektprofil (YAML/JSON)")
    parser.add_argument("--frame-export", action="store_true", help="Frames zusätzlich speichern")
    parser.add_argument("--output-frame-pattern", help="Dateimuster für Frame-Export, z.B. output/frame_{index:06d}.png")
    parser.add_argument("--config-name", help="Name des aktiven Presets")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log-Level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    preset_data = None
    if args.preset:
        preset_data = load_preset(Path(args.preset))
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config = build_config(cli_args, preset_data)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = pipeline_from_config(config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
