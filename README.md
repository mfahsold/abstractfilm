# Abstrakte Morphing-Kunstfilm Pipeline

Dieses Projekt implementiert eine lokale, modulare Videoverarbeitungspipeline, die Stadtaufnahmen in einen abstrakten Morphing-Kunstfilm verwandelt. Die Pipeline kombiniert Shot-basierte Verarbeitung, dichten optischen Fluss, ModernGL-Shaderfeedback und retro-neonfarbiges Color-Grading.

## Features
- Shot-Erkennung mit PySceneDetect und konfigurierbaren Szeneparametern
- Optical-Flow-gestütztes Morphing (Farneback oder DIS) inklusive Remapping-Blend
- GPU-Feedback-Loops via ModernGL (Reaction-Diffusion, Noise, Glow)
- Kapselbare Effektprofile (Preview/Full + Variant-Rotation oder eigene YAML/JSON-Profile)
- Farbkomposition mit LUT-Unterstützung, Retro-Neon-Gradings und Analog-Overlays (Grain, Flicker, Vignette)
- Gray-Scott-Reaktionsdiffusion mit optischem Fluss als Advektionsfeld für lebendige Muster
- CLI-basierte Workflows für Preview- und Produktions-Renderings

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --help
```

## Projektstruktur
```
assets/              # Eingangsmedien, LUTs
output/              # Render-Ergebnisse
preview_configs/     # JSON-Presets für schnelle Iterationen
effects_profiles/    # YAML-Profile für Reaction-Diffusion/Feedback/Grading
shaders/             # GLSL Shader (Fragment, Vertex)
src/                 # Python Modules der Pipeline
tests/               # Automatisierte Tests
```

Weitere Details zu Produktzielen und Anforderungen siehe `Product Requirements Document.md`.

## Typische Kommandos

Preview mit integrierten Defaults:
```bash
source .venv/bin/activate
python -m src.main --input assets/video.mp4 --output output/preview.mp4 --preview --scene-detect
```

Produktionslauf mit voller Auflösung:
```bash
python -m src.main --input assets/video.mp4 --output output/final.mp4 --full --scene-detect --include-audio
```

Eigenes Effektprofil anschließen:
```bash
python -m src.main --input assets/video.mp4 --output output/alt_preview.mp4 --preview --effects-profile effects_profiles/custom.yaml
```

Filmkonfiguration (Gray-Scott + Variantenmix) über Preset:
```bash
python -m src.main --input assets/video.mp4 --output output/final.mp4 --preset preview_configs/film_full.yaml
```
