# Product Requirements Document - Lokale Morphing-Kunstfilm Pipeline

## Dokumentinformationen
- Version: 0.1
- Stand: 2025-09-18
- Autor: Codex (Arbeitsentwurf)
- Status: Draft

## 1. Vision & Zielsetzung
### Produktvision
Eine lokale, reproduzierbare Processing-Pipeline transformiert reale Stadtaufnahmen in einen abstrakten Morphing-Kunstfilm. Optischer Fluss, organische Shader-Feedback-Loops und retro-neonfarbiges Grading verschmelzen zu einer eigenständigen audiovisuellen Handschrift, ohne auf Cloud-Ressourcen oder proprietäre Tools angewiesen zu sein.

### Ziele (Must Have)
- Offline-Pipeline, die auf handelsüblicher CPU/GPU-Hardware läuft und Frame-für-Frame arbeitet.
- Shot-basierte Verarbeitung mit automatischer Schnitt-Erkennung und konfigurierbaren Effektparametern je Szene.
- Dichte optische Flussberechnung (Farneback oder DIS) und Remapping, um organische Morphing-Übergänge zu erzeugen.
- ModernGL-basierte Shader-Effekte (Feedback, Noise, Reaction-Diffusion) inklusive Ping-Pong-Rendering.
- Farbkomposition im retro-neon-pastell Stil, inkl. Glow-Handling und LUT-Unterstützung.
- Export eines H.264 (MP4) Endprodukts, optional mit Originalaudio.

### Optionale Ziele (Nice to Have)
- Presets für schnelle Previews (niedrige Auflösung, reduzierte Iterationen).
- GUI-loses Monitoring (z. B. CLI-Progressbars, Logging) sowie Config-Dateien pro Shot.
- Austauschbare Shader-Module und LUTs für iterative Experimente.

### Nicht-Ziele
- Keine Echtzeit- oder Live-Performance-Anforderungen.
- Keine Cloud- oder Web-Deployments; Fokus auf lokale Assets.
- Audio-Design oder -Bearbeitung jenseits des einfachen Remuxens.

## 2. Stakeholder & Zielnutzer
- **Künstlerische Leitung**: definiert Look, Farbwelten, finale Abnahme.
- **Technische Künstler:innen / Creative Coders**: bauen Pipeline, entwickeln Shader, parametrieren Effekte.
- **Postproduktion**: kümmert sich um finales Encoding, Integration von Audio oder zusätzliche Grading-Pässe.

## 3. Use Cases
- **UC-1 Preview Render**: Nutzer startet 10-Sekunden-Schnellvorschau mit halber Auflösung zur Look-Iteration.
- **UC-2 Full Render**: Komplettes Quellvideo wird mit maximaler Qualität, Szenen-Resets und Audio-Mixdown verarbeitet.
- **UC-3 Shot Tuning**: Für eine identifizierte Szene werden Parameter (Optical-Flow-Intensität, Shader-Seeds, LUT) angepasst und erneut gerendert.

## 4. End-to-End Workflow (Happy Path)
1. Nutzer definiert Eingabevideo, gewünschte Ausgabedatei und Pipeline-Parameter (CLI oder Config).
2. Shot-Erkennung erzeugt Segmentlisten und setzt Feedback-Loops an Shot-Grenzen zurück.
3. Pro Shot wird Frame-für-Frame optischer Fluss berechnet, vorherige Frames morphologisch angepasst.
4. ModernGL-Shader erzeugen Feedback-/Noise-/Reaction-Diffusion-Texturen via Ping-Pong-Rendering.
5. Compositing mischt Echtbild und GPU-Texturen, globales Grading inklusive Glow wird angewandt.
6. Ergebnisframes werden encodiert, optional Audio remuxt, finales MP4 ausgegeben.
7. Logs und Metriken dokumentieren Laufzeit, Effekte und eventuelle Warnungen.

## 5. Funktionale Anforderungen
### 5.1 Input-Verarbeitung & Shot Handling
- **FR-1**: System muss Videos über OpenCV oder MoviePy einlesen können.
- **FR-2**: Bei aktivierter Option muss PySceneDetect Shots liefern, inkl. Start/End-Frames.
- **FR-3**: Pipeline muss pro Shot Feedback-Puffer und Seeds zurücksetzen können.
- **FR-4**: Szene-spezifische Parameter müssen aus Konfig-Dateien oder heuristisch (z. B. Bewegungsintensität) geladen werden können.

### 5.2 Optical Flow & Morphing
- **FR-5**: Anwender wählt zwischen Farneback und DIS (inkl. Presets) per CLI-Flag.
- **FR-6**: Flussberechnung erfolgt auf Graustufen, Ergebnisse werden als Float32 Numpy-Arrays (H x W x 2) gehalten.
- **FR-7**: Remapping via `cv2.remap` verformt vorherige Frames in Richtung aktueller Frames.
- **FR-8**: Blend-Logik (z. B. gewichtetes Addieren) ist parametrierbar, um Trail/Smear-Ausprägung zu steuern.

### 5.3 GPU-Prozedurale Effekte
- **FR-9**: ModernGL-Kontext muss headless (standalone) initialisiert werden.
- **FR-10**: Mindestens zwei Ping-Pong-Texturen (RGBA) pro Effekt-Layer werden verwaltet.
- **FR-11**: Shader-Pipeline unterstützt Feedback-, Noise- und Reaction-Diffusion-Programme.
- **FR-12**: Iterationsanzahl pro Frame (rd_steps) ist konfigurierbar, inklusive Reset an Shot-Grenzen.
- **FR-13**: Shader erhalten aktuelle Videoframes, Glow-Masken und optionale Noise Seeds als Uniforms.

### 5.4 Compositing & Farbästhetik
- **FR-14**: RD-/Noise-Outputs werden mit verformten Frames gemischt (z. B. Overlay, Multiply, Displacement).
- **FR-15**: Farb-LUTs (3D .cube) können in Shadern oder später im FFmpeg-Filter angewendet werden.
- **FR-16**: Glow-Effekt implementiert Bright-Pass + Blur + Screen/Blend, entweder GPU-intern oder per FFmpeg.
- **FR-17**: Farbkurven (Sättigung, Gamma, Schattentönung) sind parametrierbar und reproduzierbar.

### 5.5 Encoding & Ausgabe
- **FR-18**: Zwischenablage der Frames als Sequenz (PNG) oder Streaming an VideoWriter muss möglich sein.
- **FR-19**: Export nach H.264 (MP4, yuv420p) mit konfigurierbarem CRF (Default 18-20).
- **FR-20**: Optional kann Originalaudio übernommen oder stummgeschaltet werden.
- **FR-21**: Abschließende Filter (Glow, LUT) können via FFmpeg-Filtergraph ausgeführt werden, wenn nicht bereits in Shadern erfolgt.

### 5.6 Konfigurationsmanagement
- **FR-22**: CLI nimmt Parameter wie `--input`, `--output`, `--scene-detect`, `--width`, `--height`, `--flow-algo`, `--rd-steps`, `--glow`, `--use-lut` entgegen.
- **FR-23**: Preset-Dateien (JSON/YAML) in `preview_configs/` überschreiben Standardwerte.
- **FR-24**: Logging auf DEBUG/INFO-Level hält pro Frame Laufzeiten und aktive Parameter fest.

## 6. Systemarchitektur
### 6.1 Softwaremodule
- `src/main.py`: CLI, Orchestrierung, Preset-Ladung, Pipeline-Steuerung.
- `src/scene_detect.py`: Wrapper um PySceneDetect, liefert Shots inkl. Metadaten.
- `src/optical_flow.py`: Abstraktion für Farneback/DIS, parametrisiert über CLI.
- `src/morph_warp.py`: Enthält Remapping- und Blend-Funktionen.
- `src/effects_gl.py`: Initialisiert ModernGL, verwaltet Shader, Ping-Pong-Framebuffers.
- `src/color_grading.py`: LUT-Anwendung, Glow-Masken, alternative Farbanpassungen (falls GPU-Pass entfällt).
- `src/encode.py`: Verwaltet Frame-Puffer, VideoWriter oder FFmpeg-Aufrufe, Audio-Muxing.
- `shaders/*`: GLSL-Programme (fullscreen_quad.vs, feedback.fs, reaction_diffusion.fs, grading.fs, noise.fs).

### 6.2 Datenfluss pro Shot
1. `VideoReader` liefert BGR-Frame und Graustufenabbild.
2. `optical_flow` erzeugt Flow-Matrix; `morph_warp` liefert verformten Vorframe.
3. `effects_gl` erhält verformten Frame, führt RD-/Feedback-Loops aus, mischt mit Noise.
4. Ergebnis (RGBA) wird zurück auf CPU geholt oder direkt weitergereicht.
5. `color_grading` (oder Shader-Pass) finalisiert Look; `encode` speichert Frame oder streamt zu FFmpeg.

### 6.3 Zwischenspeicherung & Ressourcen
- Frame-Puffer optional auf Disk (PNG) oder im Speicher (Numpy Queue).
- Shader-States je Shot zurückgesetzt; Seed-Management via RNG-Klasse.
- GPU-Speicherbedarf: 2-3 RGBA-Texturen in Framegröße (Default 1024x576) + temporäre LUT/Noise-Texturen.

## 7. Technische Rahmenbedingungen
- **Hardware**: Desktop/Laptop mit moderner GPU (OpenGL 4.3+), 16 GB RAM empfohlen.
- **Software-Abhängigkeiten**: Python 3.10+, OpenCV, PySceneDetect, ModernGL, NumPy, MoviePy, FFmpeg CLI.
- **Plattform**: Linux primär, macOS/Windows sekundär (ModernGL muss getestet werden).
- **Performance-Ziel**: <150 ms pro Frame bei 1024x576 auf Mittelklasse-GPU für Produktionslauf.
- **Speicher**: Temporäre Sequenzspeicherung darf 100 GB nicht überschreiten; Cleanup-Skripte erforderlich.

## 8. Bedienkonzept & CLI
- Default-Preset rendert komplette Sequenz mit Farneback, 5 RD-Steps, Glow on, LUT on.
- Preview-Preset (`preview_configs/fast_preset.json`) halbiert Auflösung, reduziert RD-Steps auf 1, deaktiviert Glow.
- Fortschritt: CLI zeigt Shot- und Frame-Zähler; optional CSV-Export der Laufzeiten.
- Fehlerfälle: Graceful Shutdown bei unterbrochener Shader-Initialisierung, Logging von FFmpeg-Fehlern.

## 9. Qualitätssicherung
- Unit-Tests für Flow-Berechnung (synthetische Bewegungen) und Remap-Korrektheit.
- Shader Smoke-Tests mit Referenzbildern (Vergleich per Bilddiff, Toleranz <3%).
- End-to-End Testclip (10s) mit fixem Seed zur Regressionserkennung.
- Visuelle Review-Checkliste: Shot-Wechsel ohne Artefakte, Glow nicht ausgebrannt, Pastell-Farbwerte stabil.

## 10. Erfolgsmetriken / Akzeptanzkriterien
- Vollständiger Render eines 2-Minuten-Clips (<20 min auf Zielhardware) ohne Unterbrechung.
- Shots werden korrekt segmentiert, Feedback-Loops zeigen keine Ghosting-Artefakte.
- Abstrakte Effekte bleiben konsistent reproduzierbar bei gleichem Seed.
- Finales Video erfüllt Look-Vorgaben (Pastell, Glow) laut kreativer Leitung.

## 11. Projektplan & Meilensteine
1. **M1 - Infrastruktur (2 Wochen)**: Repo-Gerüst, Abhängigkeits-Setup, Beispielshader.
2. **M2 - Pipeline Core (3 Wochen)**: Shot-Erkennung, Optical Flow, Remap/Blend.
3. **M3 - GPU-Effekte (4 Wochen)**: Reaction-Diffusion, Feedback, Noise Integration.
4. **M4 - Grading & Encoding (2 Wochen)**: LUT/Glow, Audio-Muxing, CLI-Funktionen.
5. **M5 - QA & Feintuning (2 Wochen)**: Presets, Tests, Dokumentation, finaler Render.

## 12. Risiken & Gegenmaßnahmen
- **R1 Performance-Einbrüche**: Flow + Shader zu langsam. *Mitigation*: Presets, Downscaling, GPU-Profiling, Wechsel auf DIS.
- **R2 Shader-Stabilität**: ModernGL-Kompatibilität. *Mitigation*: Früh testen auf Zielhardware, Fallback ohne GPU-Effekte.
- **R3 Farbraumbanding**: Pastellflächen banding im H.264. *Mitigation*: 10-Bit Encoding oder dithering/Noise.
- **R4 Speicherlimit**: PNG-Sequenzen sprengen Disk. *Mitigation*: Streaming-Encoder, temporäre Ordner bereinigen.
- **R5 Toolchain-Versionen**: Library Updates brechen Pipeline. *Mitigation*: `requirements.txt` mit fixen Versionen, virtuelle Umgebungen.

## 13. Offene Fragen
- Benötigt die künstlerische Leitung interaktive Parametersteuerung während der Wiedergabe?
- Welche Mindestauflösung wird für das Endprodukt gefordert (720p, 1080p)?
- Soll ein eigenes Audio-Sounddesign erstellt werden oder bleibt der Originalton final?
- Werden weitere Effektmodule (z. B. Partikel, Edge-Detection) benötigt?

## 14. Referenzen
- PySceneDetect: https://www.scenedetect.com/
- Dense Optical Flow (OpenCV): https://www.geeksforgeeks.org/python/python-opencv-dense-optical-flow/
- Optical Flow Remap Diskussion: https://github.com/opencv/opencv/issues/11068
- Reaction-Diffusion Shader: https://www.hugi.scene.org/online/hugi34/hugi%2034%20-%20coding%20corner%20marnix%20kok%20reaction-diffusion%20texture%20generation%20on%20gpu.htm
- ModernGL Basics: https://moderngl.readthedocs.io/en/5.6.3/the_guide/basic.html
- Retro Glow in FFmpeg: https://zayne.io/articles/retro-glow-effects-with-ffmpeg
- FFmpeg Filter Doku: https://ffmpeg.org/ffmpeg-filters.html
