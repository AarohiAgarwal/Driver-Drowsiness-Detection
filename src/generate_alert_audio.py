"""
Audio Alert Generator for Driver Drowsiness Detection System.
Generates crystal-clear voice alert audio files in English, Hindi, and Punjabi
using Google Text-to-Speech (gTTS) for 100% offline real-time deployment.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows cp1252 console encoding for Hindi/Punjabi Unicode characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config

ALERT_SCRIPTS = {
    "en": {
        "text": "Warning! Driver drowsiness detected. Please take a break.",
        "lang": "en",
        "filename": "alert_en.mp3",
        "description": "English Voice Alert"
    },
    "hi": {
        "text": "चेतावनी! चालक को नींद आ रही है। कृपया गाड़ी रोकें और आराम करें।",
        "lang": "hi",
        "filename": "alert_hi.mp3",
        "description": "Hindi Voice Alert (Devanagari)"
    },
    "pa": {
        "text": "ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ।",
        "lang": "pa",
        "filename": "alert_pa.mp3",
        "description": "Punjabi Voice Alert (Gurmukhi)"
    }
}


def generate_alerts(output_dir: Path = None, force: bool = False):
    """
    Generates and saves MP3 voice alerts for English, Hindi, and Punjabi.
    """
    from gtts import gTTS

    target_dir = output_dir or (config.BASE_DIR / "data" / "alerts")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  GENERATING MULTILINGUAL DRIVER DROWSINESS VOICE ALERTS")
    print(f"  Target Directory: {target_dir}")
    print("=" * 65)

    generated_paths = {}

    for code, info in ALERT_SCRIPTS.items():
        out_path = target_dir / info["filename"]
        if out_path.exists() and not force:
            print(f"  [EXISTS] {info['description']}: {out_path.name}")
            generated_paths[code] = out_path
            continue

        print(f"  [GENERATING] {info['description']} ({code.upper()})...")
        print(f"               Text: \"{info['text']}\"")
        try:
            tts = gTTS(text=info["text"], lang=info["lang"], slow=False)
            tts.save(str(out_path))
            print(f"  [OK] Saved: {out_path.name} ({out_path.stat().st_size} bytes)")
            generated_paths[code] = out_path
        except Exception as e:
            print(f"  [ERROR] Failed to generate {code}: {e}")

    # Generate combined multilingual alert (English -> Hindi -> Punjabi)
    combined_path = target_dir / "alert_multilingual.mp3"
    if not combined_path.exists() or force:
        print("\n  [GENERATING] Combined Multilingual Voice Alert (EN + HI + PA)...")
        combined_text = (
            "Warning! Driver drowsiness detected. Please take a break. "
            "चेतावनी! चालक को नींद आ रही है। कृपया गाड़ी रोकें और आराम करें। "
            "ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ।"
        )
        try:
            # We can generate individual segments and concatenate, or use gTTS multilingual if supported
            # Since gTTS lang is per request, we can concatenate the binary MP3 frames directly!
            en_bytes = (target_dir / "alert_en.mp3").read_bytes()
            hi_bytes = (target_dir / "alert_hi.mp3").read_bytes()
            pa_bytes = (target_dir / "alert_pa.mp3").read_bytes()
            combined_path.write_bytes(en_bytes + hi_bytes + pa_bytes)
            print(f"  [OK] Combined Audio Saved: {combined_path.name} ({combined_path.stat().st_size} bytes)")
            generated_paths["all"] = combined_path
        except Exception as e:
            print(f"  [NOTICE] Combined audio error: {e}")

    print("\nMultilingual alert generation complete!\n")
    return generated_paths


if __name__ == "__main__":
    generate_alerts(force=True)
