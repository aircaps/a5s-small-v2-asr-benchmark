from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os

import numpy as np
import google.auth
from google.auth.transport.requests import Request
from scipy.signal import resample_poly
from websockets.asyncio.client import connect

from a5sv2_eval.common import parser, realtime_chunks, run_from_args


class Gemini:
    system_id = "google_gemini_3_5_transcribe_live"
    system = "Google Gemini 3.5 Transcribe Live"
    model = "gemini-3.5-transcribe-live"
    sample_rate = 16000
    session_seconds = 540

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.access_token = os.getenv("GEMINI_ACCESS_TOKEN")
        self.language_codes = [os.getenv("GEMINI_LANGUAGE", "en-US")]
        self.drain_seconds = float(os.getenv("GEMINI_DRAIN_SECONDS", "30"))
        self.auth_lock = asyncio.Lock()
        if not self.api_key and not self.access_token:
            self.credentials, default_project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.project = os.getenv("GOOGLE_CLOUD_PROJECT") or default_project
            if not self.project:
                raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT with ADC")
        elif not self.api_key:
            self.credentials = None
            self.project = os.environ["GOOGLE_CLOUD_PROJECT"]

    def streaming_config(self, row: dict) -> dict:
        return {
            "model": self.model,
            "source_sample_rate": row["sample_rate"],
            "sample_rate": self.sample_rate,
            "audio_format": "pcm_s16le_mono",
            "chunk_ms": 100,
            "session_audio_seconds": self.session_seconds,
            "transcription_mode": "VERBATIM",
            "language_codes": self.language_codes,
            "post_audio_idle_seconds": self.drain_seconds,
        }

    def capacity_units(self, row: dict) -> int:
        return max(1, math.ceil(row["duration_seconds"] / self.session_seconds))

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        source_rate = config["source_sample_rate"]
        if source_rate == self.sample_rate:
            audio = bytes(pcm)
        else:
            samples = np.frombuffer(pcm, dtype="<i2")
            audio = (
                np.rint(resample_poly(samples, self.sample_rate, source_rate))
                .clip(-32768, 32767)
                .astype("<i2")
                .tobytes()
            )
        size = self.session_seconds * self.sample_rate * 2
        intervals = [(start, min(start + size, len(audio))) for start in range(0, len(audio), size)]
        sessions = await asyncio.gather(
            *(self._transcribe_session(audio[start:end], config) for start, end in intervals)
        )
        return {
            "text": " ".join(session.pop("text").strip() for session in sessions).strip(),
            "sessions": [
                {
                    **session,
                    "source_start_seconds": start / (self.sample_rate * 2),
                    "source_end_seconds": end / (self.sample_rate * 2),
                }
                for session, (start, end) in zip(sessions, intervals)
            ],
            "streamed_audio_seconds": len(audio) / (self.sample_rate * 2),
            "streamed_pcm_sha256": hashlib.sha256(audio).hexdigest(),
        }

    async def _transcribe_session(self, audio: bytes, config: dict) -> dict:
        base_url = (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        )
        headers = None
        if self.api_key:
            url = f"{base_url}?key={self.api_key}"
        else:
            if self.credentials:
                async with self.auth_lock:
                    if not self.credentials.valid:
                        await asyncio.to_thread(self.credentials.refresh, Request())
                self.access_token = self.credentials.token
            url = base_url
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "x-goog-user-project": self.project,
            }
        finals, metadata = [], {
            "server_messages": 0,
            "final_events": 0,
            "interim_events": 0,
            "turn_complete_events": 0,
        }
        try:
            async with connect(
                url,
                open_timeout=30,
                close_timeout=10,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
                additional_headers=headers,
            ) as websocket:
                await websocket.send(
                    json.dumps(
                        {
                            "setup": {
                                "model": f"models/{self.model}",
                                "generationConfig": {"responseModalities": ["TEXT"]},
                                "inputAudioTranscription": {
                                    "languageCodes": config["language_codes"],
                                    "mode": config["transcription_mode"],
                                },
                            }
                        }
                    )
                )
                acknowledgement = json.loads(await asyncio.wait_for(websocket.recv(), 30))
                if "setupComplete" not in acknowledgement:
                    raise RuntimeError(f"setup failed: {acknowledgement}")
                audio_done = asyncio.Event()
                clock = asyncio.get_running_loop()
                audio_done_at = None
                last_message_at = None

                async def receive() -> None:
                    nonlocal last_message_at
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                websocket.recv(), config["post_audio_idle_seconds"]
                                if audio_done.is_set()
                                else 5
                            )
                        except asyncio.TimeoutError:
                            if audio_done.is_set():
                                return
                            continue
                        last_message_at = clock.time()
                        metadata["server_messages"] += 1
                        message = json.loads(raw)
                        if "goAway" in message:
                            metadata["go_away"] = message["goAway"]
                        content = message.get("serverContent", {})
                        interim = content.get("interimInputTranscription")
                        final = content.get("inputTranscription")
                        if content.get("turnComplete"):
                            metadata["turn_complete_events"] += 1
                        if interim:
                            metadata["interim_events"] += 1
                        if final:
                            finals.append(final.get("text", ""))
                            metadata["final_events"] += 1

                receiver = asyncio.create_task(receive())
                try:
                    chunk_bytes = self.sample_rate * 2 * config["chunk_ms"] // 1000
                    async for chunk in realtime_chunks(audio, self.sample_rate, chunk_bytes):
                        await websocket.send(
                            json.dumps(
                                {
                                    "realtimeInput": {
                                        "audio": {
                                            "data": base64.b64encode(chunk).decode(),
                                            "mimeType": "audio/pcm;rate=16000",
                                        }
                                    }
                                }
                            )
                        )
                    await websocket.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                    audio_done_at = clock.time()
                    audio_done.set()
                    await asyncio.wait_for(receiver, config["post_audio_idle_seconds"] + 15)
                finally:
                    if not receiver.done():
                        receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
        except Exception as error:
            message = str(error)
            if self.api_key:
                message = message.replace(self.api_key, "<redacted>")
            raise RuntimeError(message) from None

        if audio_done_at is not None:
            metadata["final_response_delay_seconds"] = max(
                0.0, (last_message_at or audio_done_at) - audio_done_at
            )
        return {
            "text": " ".join(text.strip() for text in finals).strip(),
            **metadata,
            "empty_transcript": not any(text.strip() for text in finals),
            "streamed_pcm_sha256": hashlib.sha256(audio).hexdigest(),
            "streamed_audio_seconds": len(audio) / (self.sample_rate * 2),
        }


def main() -> None:
    args = parser(
        "Stream the benchmark through Gemini 3.5 Transcribe Live",
        "google_gemini_3_5_transcribe_live.jsonl",
        concurrency=int(os.getenv("GEMINI_CONCURRENCY", "50")),
    ).parse_args()
    run_from_args(Gemini, args)


if __name__ == "__main__":
    main()
