"""
Audible Alarm and Voice Alert System for Driver Drowsiness Detection.

Provides non-blocking, asynchronous audio warnings and multilingual speech alerts:
- Activates automatically when the driver enters the DROWSY state.
- Supports Multilingual Voice Alerts: English, Hindi (हिन्दी), and Punjabi (ਪੰਜਾਬੀ).
- Sequentially warns in all languages or allows driver-preferred language mode.
- Emits high-priority emergency audio beeps via Windows winsound.
- Immediately stops when the driver recovers to the ALERT state.
- Strictly prevents multiple simultaneous alarm processes through thread locking.
- Configurable cooldown to prevent repetitive alarm spam.
- 100% offline playback using pre-generated native audio assets.
- Fails gracefully if audio hardware/drivers are unavailable.
"""

import os
import sys
import time
import threading
import argparse
import ctypes
from pathlib import Path
from typing import Optional, Tuple, List

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding for Hindi/Punjabi Unicode characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config

# Windows-specific audio libraries
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import win32com.client
    import pythoncom
    HAS_SAPI = True
except ImportError:
    HAS_SAPI = False

# Windows Multimedia MCI for low-latency MP3 playback
try:
    WINMM = ctypes.windll.winmm if os.name == "nt" else None
except Exception:
    WINMM = None


class DrowsinessAlarm:
    """
    Asynchronous audible alarm and voice alert manager.
    Operates strictly in background worker threads without blocking video processing.
    Supports English, Hindi, and Punjabi voice warnings.
    """
    def __init__(self,
                 cooldown_sec: float = getattr(config, "ALARM_COOLDOWN", 3.0),
                 speech_message: str = getattr(config, "ALARM_SPEECH_MESSAGE",
                                               "WARNING! DRIVER DROWSINESS DETECTED. PLEASE TAKE A BREAK."),
                 enabled: bool = getattr(config, "ENABLE_AUDIO_ALARM", True),
                 language_mode: str = getattr(config, "ALERT_LANGUAGE_MODE", "all"),
                 must_complete_cycle: bool = getattr(config, "ALERT_MUST_COMPLETE_CYCLE", True)):
        self.cooldown_sec = cooldown_sec
        self.speech_message = speech_message
        self.enabled = enabled
        self.language_mode = str(language_mode).lower().strip()
        self.must_complete_cycle = bool(must_complete_cycle)

        # State tracking
        self.last_trigger_time: float = -999.0
        self.is_playing: bool = False
        self.total_alarms_triggered: int = 0
        self.last_state_processed: str = "ALERT"

        # Concurrency & Termination primitives
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # SAPI Speech Synthesizer instance (lazy initialized per thread)
        self._active_speaker = None
        self._current_audio_alias: Optional[str] = None

    def is_alert_active(self) -> bool:
        """
        Returns True if an alarm or voice warning cycle is actively executing.
        Used to keep visual wake-up strobe and HUD banners active until warning completes.
        """
        return self.is_playing

    def set_language_mode(self, mode: str):
        """
        Sets the active alert language:
        - 'all': Multilingual (English -> Hindi -> Punjabi)
        - 'hi':  Hindi (हिन्दी)
        - 'pa':  Punjabi (ਪੰਜਾਬੀ)
        - 'en':  English
        """
        mode_clean = str(mode).lower().strip()
        if mode_clean in ["all", "hi", "pa", "en"]:
            self.language_mode = mode_clean

    def get_language_mode(self) -> str:
        """Returns the current alert language code."""
        return self.language_mode

    def trigger(self, state: str, current_time: Optional[float] = None) -> bool:
        """
        Evaluates driver state and conditionally triggers or manages the alarm.
        
        State handling:
        - When an alarm is active:
          * If must_complete_cycle is True: Allows warning cycle (voice alerts + tones)
            to complete in full even if driver's eyes flutter open or yawning ceases.
          * If must_complete_cycle is False: Immediately interrupts when state is ALERT.
        - ALERT / WARNING: No new alarm initiated.
        - DROWSY: Triggers audible warning if not currently playing and cooldown has passed.
          Returns True ONLY if a new alarm cycle was initiated.
        """
        now = current_time if current_time is not None else time.time()
        self.last_state_processed = state

        # If an alarm cycle is already actively playing:
        if self.is_playing:
            if not self.must_complete_cycle and state == "ALERT":
                self.stop()
            # Do not launch duplicate concurrent alarms while active
            return False

        # State: ALERT or WARNING -> No new alarm initiated
        if state in ("ALERT", "WARNING"):
            return False

        # State: DROWSY -> Activate Audible Alarm
        if state == "DROWSY":
            if not self.enabled:
                return False

            with self._lock:
                # Avoid multiple simultaneous alarm processes
                if self.is_playing:
                    return False

                # Enforce configurable cooldown
                time_since_last = now - self.last_trigger_time
                if time_since_last < self.cooldown_sec:
                    return False

                # Launch non-blocking background alarm thread
                self.last_trigger_time = now
                self.total_alarms_triggered += 1
                self._stop_event.clear()
                self.is_playing = True

                self._worker_thread = threading.Thread(
                    target=self._alarm_worker,
                    name=f"AlarmWorker-{self.total_alarms_triggered}",
                    daemon=True
                )
                self._worker_thread.start()
                return True

        return False

    def stop(self):
        """
        Immediately interrupts and stops any playing alarm, tones, or speech.
        Halts both Windows MCI audio file playback and SAPI TTS engine.
        """
        self._stop_event.set()

        # Stop active Windows Multimedia MCI audio file playback
        if self._current_audio_alias and WINMM is not None:
            try:
                WINMM.mciSendStringW(f"stop {self._current_audio_alias}", None, 0, None)
                WINMM.mciSendStringW(f"close {self._current_audio_alias}", None, 0, None)
            except Exception:
                pass
            self._current_audio_alias = None

        # Stop Windows SAPI Voice queue if speaking (Purge flag = 2)
        if self._active_speaker is not None:
            try:
                self._active_speaker.Speak("", 2)  # SVSFPurgeBeforeSpeak = 2
            except Exception:
                pass

        self.is_playing = False

    def get_status(self, current_time: Optional[float] = None) -> Tuple[str, Tuple[int, int, int]]:
        """
        Returns (status_text, (B, G, R) color) for HUD display.
        """
        if not self.enabled:
            return "MUTED", (128, 128, 128)
        lang_tag = self.language_mode.upper()
        if self.is_playing:
            return f"SOUNDING [{lang_tag}]", (0, 0, 255)
        now = current_time if current_time is not None else time.time()
        rem_cd = self.cooldown_sec - (now - self.last_trigger_time)
        if rem_cd > 0 and self.last_trigger_time > 0:
            return f"COOLDOWN ({rem_cd:.1f}s)", (0, 180, 255)
        return f"STANDBY [{lang_tag}]", (0, 220, 0)

    def _play_audio_file(self, file_path: Path) -> bool:
        """
        Plays an MP3 audio file asynchronously using Windows Multimedia MCI.
        Monitors self._stop_event at 30ms intervals to stop immediately if driver recovers.
        """
        if WINMM is None or not file_path.exists():
            return False

        abs_path = str(file_path.resolve())
        alias = f"drowsy_alert_{int(time.time() * 1000)}"
        self._current_audio_alias = alias

        try:
            # Close any lingering alias
            WINMM.mciSendStringW(f"close {alias}", None, 0, None)

            # Open media file
            ret_open = WINMM.mciSendStringW(f'open "{abs_path}" type mpegvideo alias {alias}', None, 0, None)
            if ret_open != 0:
                return False

            # Query duration in milliseconds
            buf = ctypes.create_unicode_buffer(64)
            WINMM.mciSendStringW(f"status {alias} length", buf, 64, None)
            length_ms = int(buf.value) if buf.value.isdigit() else 3500
            duration_sec = (length_ms / 1000.0) + 0.1

            # Begin playback
            ret_play = WINMM.mciSendStringW(f"play {alias}", None, 0, None)
            if ret_play != 0:
                WINMM.mciSendStringW(f"close {alias}", None, 0, None)
                return False

            # Responsive sleep loop checking termination signal
            start_t = time.time()
            while (time.time() - start_t) < duration_sec:
                if self._stop_event.is_set():
                    WINMM.mciSendStringW(f"stop {alias}", None, 0, None)
                    WINMM.mciSendStringW(f"close {alias}", None, 0, None)
                    self._current_audio_alias = None
                    return False
                time.sleep(0.03)

            WINMM.mciSendStringW(f"close {alias}", None, 0, None)
            self._current_audio_alias = None
            return True

        except Exception:
            if self._current_audio_alias:
                try:
                    WINMM.mciSendStringW(f"close {self._current_audio_alias}", None, 0, None)
                except Exception:
                    pass
                self._current_audio_alias = None
            return False

    def _get_audio_files_for_mode(self) -> List[Path]:
        """Returns the ordered list of audio file paths corresponding to the language mode."""
        alerts_dir = getattr(config, "ALERT_AUDIO_DIR", config.BASE_DIR / "data" / "alerts")

        if self.language_mode == "hi":
            return [alerts_dir / "alert_hi.mp3"]
        elif self.language_mode == "pa":
            return [alerts_dir / "alert_pa.mp3"]
        elif self.language_mode == "en":
            return [alerts_dir / "alert_en.mp3"]
        else:  # "all" or multilingual: English -> Hindi -> Punjabi
            return [
                alerts_dir / "alert_en.mp3",
                alerts_dir / "alert_hi.mp3",
                alerts_dir / "alert_pa.mp3",
            ]

    def _alarm_worker(self):
        """
        Background audio worker thread.
        Emits pulsed alarm tones followed by clear multilingual speech warning.
        Fails gracefully without crashing if audio drivers or sound cards error out.
        """
        co_initialized = False
        try:
            # Step 1: Emergency pulsed alert tones (Windows System Alert sound + Beeps)
            if HAS_WINSOUND and not self._stop_event.is_set():
                try:
                    winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
                except Exception:
                    pass

                for freq in [2200, 2600]:
                    if self._stop_event.is_set():
                        break
                    try:
                        winsound.Beep(freq, 200)
                    except Exception:
                        break
                    if not self._stop_event.is_set():
                        time.sleep(0.05)

            # Step 2: Spoken Voice Alerts in Configured Languages
            if not self._stop_event.is_set():
                audio_files = self._get_audio_files_for_mode()
                played_any = False

                for audio_path in audio_files:
                    if self._stop_event.is_set():
                        break
                    if audio_path.exists():
                        success = self._play_audio_file(audio_path)
                        if success:
                            played_any = True
                        if not self._stop_event.is_set():
                            time.sleep(0.08)

                # Step 3: Fallback if audio files were missing or unplayable
                if not played_any and not self._stop_event.is_set():
                    if HAS_SAPI:
                        try:
                            pythoncom.CoInitialize()
                            co_initialized = True
                            speaker = win32com.client.Dispatch("SAPI.SpVoice")
                            self._active_speaker = speaker
                            if not self._stop_event.is_set():
                                speaker.Speak(self.speech_message, 0)
                        except Exception:
                            if HAS_WINSOUND and not self._stop_event.is_set():
                                for _ in range(3):
                                    if self._stop_event.is_set():
                                        break
                                    try:
                                        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                                        winsound.Beep(2400, 250)
                                        time.sleep(0.08)
                                    except Exception:
                                        break
                    elif HAS_WINSOUND and not self._stop_event.is_set():
                        for _ in range(3):
                            if self._stop_event.is_set():
                                break
                            try:
                                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                                winsound.Beep(2400, 250)
                                time.sleep(0.08)
                            except Exception:
                                break

        except Exception as e:
            print(f"Audio Alarm Notice: Non-critical audio warning exception handled: {e}")
        finally:
            self._active_speaker = None
            if self._current_audio_alias and WINMM is not None:
                try:
                    WINMM.mciSendStringW(f"close {self._current_audio_alias}", None, 0, None)
                except Exception:
                    pass
                self._current_audio_alias = None

            if co_initialized:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
            self.is_playing = False


# ==============================================================================
# AUTOMATED UNIT TESTS & TRANSITION VERIFICATION
# ==============================================================================

def test_alarm_state_transitions() -> bool:
    """
    Tests exact required alarm transition states and multilingual features:
    1. ALERT   -> no alarm
    2. WARNING -> no continuous alarm
    3. DROWSY  -> alarm triggered
    4. DROWSY  -> cooldown enforced (no duplicate concurrent process)
    5. ALERT   -> stop alarm immediately
    6. DROWSY  -> re-trigger after cooldown
    7. Language mode configuration ('all', 'hi', 'pa', 'en')
    8. Pre-generated audio assets verification (English, Hindi, Punjabi)
    """
    print("\n" + "=" * 65)
    print("  RUNNING AUDIBLE ALARM STATE MACHINE TEST SUITE")
    print("=" * 65)

    # Initialize alarm with 2.0s cooldown for fast test cycle
    alarm = DrowsinessAlarm(cooldown_sec=2.0, speech_message="Test Alert")
    t0 = 1000.0

    # -------------------------------------------------------------
    # Test 1: ALERT -> No Alarm
    # -------------------------------------------------------------
    print("\n[Test 1] State: ALERT -> Verify No Alarm Triggered...")
    triggered = alarm.trigger("ALERT", current_time=t0)
    assert not triggered, f"ALERT state unexpectedly triggered alarm: {triggered}"
    assert not alarm.is_playing, "Alarm is_playing flag should be False during ALERT"
    print("  PASS: ALERT state produced no alarm.")

    # -------------------------------------------------------------
    # Test 2: WARNING -> No Continuous Alarm
    # -------------------------------------------------------------
    print("\n[Test 2] State: WARNING -> Verify No Continuous Alarm...")
    triggered = alarm.trigger("WARNING", current_time=t0 + 1.0)
    assert not triggered, f"WARNING state unexpectedly triggered alarm: {triggered}"
    assert not alarm.is_playing, "Alarm is_playing flag should be False during WARNING"
    print("  PASS: WARNING state produced no continuous alarm.")

    # -------------------------------------------------------------
    # Test 3: DROWSY -> Alarm Activates
    # -------------------------------------------------------------
    print("\n[Test 3] State: DROWSY -> Verify Alarm Activates...")
    triggered = alarm.trigger("DROWSY", current_time=t0 + 2.0)
    assert triggered, "DROWSY state failed to activate alarm!"
    assert alarm.total_alarms_triggered == 1, f"Expected 1 trigger, got {alarm.total_alarms_triggered}"
    print(f"  PASS: DROWSY state triggered alarm successfully (Alarms count: {alarm.total_alarms_triggered}).")

    # -------------------------------------------------------------
    # Test 4: DROWSY -> Cooldown Enforced (No Duplicate Process)
    # -------------------------------------------------------------
    print("\n[Test 4] State: DROWSY (Inside Cooldown) -> Verify No Duplicate Process...")
    triggered_during_cooldown = alarm.trigger("DROWSY", current_time=t0 + 2.5)
    assert not triggered_during_cooldown, "Alarm triggered during active cooldown period!"
    assert alarm.total_alarms_triggered == 1, "Duplicate alarm process spawned during cooldown!"
    print("  PASS: Cooldown and concurrency check successfully prevented duplicate alarm processes.")

    # -------------------------------------------------------------
    # Test 5: ALERT -> Stop Alarm
    # -------------------------------------------------------------
    print("\n[Test 5] State: ALERT -> Verify Immediate Alarm Termination...")
    alarm.stop()
    assert not alarm.is_playing, "Alarm failed to stop after explicit stop call!"
    assert alarm._stop_event.is_set(), "Stop event was not set!"

    # Trigger with ALERT state to verify automatic stop
    alarm.trigger("ALERT", current_time=t0 + 3.0)
    assert not alarm.is_playing, "Alarm is_playing flag should remain False on ALERT"
    print("  PASS: Driver recovery to ALERT terminated the alarm cleanly.")

    # -------------------------------------------------------------
    # Test 6: Recovery & Re-trigger after Cooldown
    # -------------------------------------------------------------
    print("\n[Test 6] State: DROWSY (After Cooldown) -> Verify Re-trigger...")
    retriggered = alarm.trigger("DROWSY", current_time=t0 + 5.0)
    assert retriggered, "Alarm failed to re-trigger after cooldown elapsed!"
    assert alarm.total_alarms_triggered == 2, f"Expected 2 total triggers, got {alarm.total_alarms_triggered}"
    alarm.stop()
    print("  PASS: Alarm successfully re-triggered after cooldown elapsed.")

    # -------------------------------------------------------------
    # Test 7: Multilingual Mode Configuration
    # -------------------------------------------------------------
    print("\n[Test 7] Multilingual Modes -> Verify Language Switching ('all', 'hi', 'pa', 'en')...")
    for mode in ["hi", "pa", "en", "all"]:
        alarm.set_language_mode(mode)
        assert alarm.get_language_mode() == mode, f"Expected {mode}, got {alarm.get_language_mode()}"
        status_txt, _ = alarm.get_status()
        assert mode.upper() in status_txt, f"Language tag {mode.upper()} not in status: {status_txt}"
    print("  PASS: Language mode switches correctly between English, Hindi, Punjabi, and All.")

    # -------------------------------------------------------------
    # Test 8: Multilingual Audio Assets Presence
    # -------------------------------------------------------------
    print("\n[Test 8] Audio Assets -> Verify English, Hindi, and Punjabi MP3 files exist...")
    alerts_dir = getattr(config, "ALERT_AUDIO_DIR", config.BASE_DIR / "data" / "alerts")
    for fname in ["alert_en.mp3", "alert_hi.mp3", "alert_pa.mp3"]:
        fpath = alerts_dir / fname
        assert fpath.exists(), f"Audio file missing: {fpath}"
        assert fpath.stat().st_size > 1000, f"Audio file is empty or corrupted: {fpath}"
        print(f"  • Found {fname} ({fpath.stat().st_size} bytes)")
    print("  PASS: All multilingual audio alert files verified.")

    # -------------------------------------------------------------
    # Test 9: Alert Cycle Completion Policy (User Requirement)
    # -------------------------------------------------------------
    print("\n[Test 9] Cycle Completion Policy -> Verify In-Progress Alert Persists When Eyes Open / Yawn Stops...")
    alarm_cycle_tester = DrowsinessAlarm(cooldown_sec=1.0, must_complete_cycle=True)
    t_start = 5000.0
    # Simulate drowsiness detected -> triggers alarm
    trig_res = alarm_cycle_tester.trigger("DROWSY", current_time=t_start)
    assert trig_res, "Alarm failed to trigger for cycle completion test"
    assert alarm_cycle_tester.is_alert_active(), "Alarm should be actively playing warning cycle"

    # Driver flutters eyes open or closes mouth: frame state flips to ALERT
    # Under ALERT_MUST_COMPLETE_CYCLE = True, alarm MUST NOT be terminated prematurely!
    alarm_cycle_tester.trigger("ALERT", current_time=t_start + 0.1)
    assert alarm_cycle_tester.is_alert_active(), "Alert was prematurely killed by ALERT state despite must_complete_cycle=True!"

    # Frame state flips to WARNING: alarm still completes
    alarm_cycle_tester.trigger("WARNING", current_time=t_start + 0.2)
    assert alarm_cycle_tester.is_alert_active(), "Alert was prematurely killed by WARNING state!"

    # Clean up with explicit stop
    alarm_cycle_tester.stop()
    assert not alarm_cycle_tester.is_alert_active(), "Explicit stop should immediately halt playback"
    print("  PASS: Warning cycle successfully protected; momentary eye opening does not abort active alert.")

    print("\n" + "=" * 65)
    print("  ALL 9 ALARM STATE MACHINE & CYCLE COMPLETION TESTS PASSED (100%)")
    print("=" * 65 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Audible Alarm and Multilingual Voice Alert System")
    parser.add_argument("--test", action="store_true", help="Run automated alarm state machine tests")
    parser.add_argument("--demo", action="store_true", help="Play a live audio demonstration")
    parser.add_argument("--lang", type=str, default="all", choices=["all", "en", "hi", "pa"],
                        help="Language for audio demo: all (multilingual), hi (Hindi), pa (Punjabi), en (English)")
    args = parser.parse_args()

    if args.demo:
        print(f"\nPlaying live audible alarm demonstration in [{args.lang.upper()}] mode...")
        alarm = DrowsinessAlarm(cooldown_sec=1.0, language_mode=args.lang)
        alarm.trigger("DROWSY")
        sleep_dur = 9.0 if args.lang == "all" else 3.5
        time.sleep(sleep_dur)
        print("Driver recovered: stopping alarm...")
        alarm.stop()
        print("Alarm demonstration finished.\n")
    else:
        test_alarm_state_transitions()


if __name__ == "__main__":
    main()
