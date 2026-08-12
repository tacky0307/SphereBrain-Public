from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread, Lock
from typing import Callable
import tempfile
import time
import wave

import numpy as np


@dataclass
class AudioStatus:
    running: bool = False
    state: str = "停止中"
    last_text: str = ""
    last_error: str = ""
    chunks_processed: int = 0
    chunks_skipped: int = 0


class SystemAudioListener:
    """
    Windowsの既定スピーカーのループバック音声を録音し、
    faster-whisperで文字化する。

    一時WAVは文字化後に削除する。
    """

    def __init__(
        self,
        on_text: Callable[[str], None],
        model_size: str = "small",
        language: str = "ja",
        chunk_seconds: int = 18,
        sample_rate: int = 48000,
        silence_rms: float = 0.006,
    ) -> None:
        self.on_text = on_text
        self.model_size = model_size
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.sample_rate = sample_rate
        self.silence_rms = silence_rms

        self.status = AudioStatus()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._model = None

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.status.running = False
        self.status.state = "停止中"

    def _load_model(self):
        if self._model is not None:
            return self._model

        self.status.state = "音声認識モデルを準備中"
        from faster_whisper import WhisperModel

        # CPU向け。GPU設定なしでも動作する。
        self._model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=1,
        )
        return self._model

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]

        # 2chを超える場合は先頭2chに制限
        if audio.shape[1] > 2:
            audio = audio[:, :2]

        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(pcm.shape[1])
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())

    def _transcribe(self, wav_path: Path) -> str:
        model = self._load_model()
        segments, _ = model.transcribe(
            str(wav_path),
            language=self.language,
            vad_filter=True,
            beam_size=3,
            condition_on_previous_text=False,
        )
        parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    def _run(self) -> None:
        self.status.running = True
        self.status.last_error = ""

        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            if speaker is None:
                raise RuntimeError("既定のスピーカーが見つかりません。")

            loopback = sc.get_microphone(
                id=str(speaker.name),
                include_loopback=True,
            )
            if loopback is None:
                raise RuntimeError("スピーカーのループバック入力が見つかりません。")

            self._load_model()
            self.status.state = f"聴取中：{speaker.name}"

            frame_count = self.sample_rate * self.chunk_seconds

            with loopback.recorder(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=2048,
            ) as recorder:
                while not self._stop_event.is_set():
                    self.status.state = f"聴取中：{speaker.name}"
                    audio = recorder.record(numframes=frame_count)

                    if self._stop_event.is_set():
                        break

                    rms = float(np.sqrt(np.mean(np.square(audio))))
                    if not np.isfinite(rms) or rms < self.silence_rms:
                        self.status.chunks_skipped += 1
                        self.status.state = "無音を待っています"
                        continue

                    self.status.state = "音声を文字化中"
                    temp_path: Path | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".wav",
                            delete=False,
                        ) as tmp:
                            temp_path = Path(tmp.name)

                        self._write_wav(temp_path, audio, self.sample_rate)
                        text = self._transcribe(temp_path)

                        if text:
                            self.status.last_text = text[:500]
                            self.status.chunks_processed += 1
                            self.on_text(text)
                    finally:
                        if temp_path and temp_path.exists():
                            try:
                                temp_path.unlink()
                            except OSError:
                                pass

        except Exception as exc:
            self.status.last_error = str(exc)
            self.status.state = "エラー"
        finally:
            self.status.running = False
