from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid

import numpy as np
from scipy.signal import resample_poly
from websockets.asyncio.client import connect

from a5sv2_eval.common import parser, realtime_chunks, run_from_args


class MetaMuse:
    system_id = "meta_muse_transcribe"
    system = "Meta Muse Transcribe"
    model = "muse-voice-transcribe-1.0"
    max_audio_seconds = 3600
    long_session_splits = {"TS3007b": [1481.75, 2994.75]}

    def __init__(self) -> None:
        self.api_key = os.environ["META_API_KEY"]
        self.mode = os.getenv("META_MODE", "ENDPOINTING")

    def streaming_config(self, row: dict) -> dict:
        sample_rate = row["sample_rate"] if row["sample_rate"] in {16000, 24000} else 24000
        config = {
            "model": self.model,
            "mode": self.mode,
            "audioEncoding": f"PCM_{sample_rate // 1000}KHZ",
            "partialMode": "CUMULATIVE",
            "languageBias": ["English"],
            "emitAudioProgress": True,
            "source_sample_rate": row["sample_rate"],
            "sample_rate": sample_rate,
            "chunk_ms": 100,
        }
        if self.mode == "PUSH_TO_TALK" and row.get("id") in self.long_session_splits:
            config["split_seconds"] = self.long_session_splits[row["id"]]
        return config

    def capacity_units(self, row: dict) -> int:
        return (
            len(self.long_session_splits.get(row.get("id"), [])) + 1
            if self.mode == "PUSH_TO_TALK"
            else 1
        )

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        source_rate, sample_rate = config["source_sample_rate"], config["sample_rate"]
        if source_rate == sample_rate:
            audio = bytes(pcm)
        else:
            samples = np.frombuffer(pcm, dtype="<i2")
            resampled = resample_poly(samples, sample_rate, source_rate)
            audio = np.rint(resampled).clip(-32768, 32767).astype("<i2").tobytes()

        split_seconds = config.get("split_seconds")
        if not split_seconds:
            return await self._transcribe_session(audio, config)

        boundaries = [0, *(int(value * sample_rate * 2) for value in split_seconds), len(audio)]
        intervals = list(zip(boundaries, boundaries[1:]))
        sessions = await asyncio.gather(
            *(self._transcribe_session(audio[start:end], config) for start, end in intervals)
        )
        text = " ".join(session.pop("text").strip() for session in sessions).strip()
        return {
            "text": text,
            "sessions": [
                {
                    **session,
                    "source_start_seconds": start / (sample_rate * 2),
                    "source_end_seconds": end / (sample_rate * 2),
                }
                for session, (start, end) in zip(sessions, intervals)
            ],
            "streamed_audio_seconds": sum(end - start for start, end in intervals)
            / (sample_rate * 2),
            "streamed_pcm_sha256": hashlib.sha256(audio).hexdigest(),
        }

    async def _transcribe_session(self, audio: bytes, config: dict) -> dict:
        sample_rate = config["sample_rate"]
        handshake = {
            key: value
            for key, value in config.items()
            if key
            not in {
                "source_sample_rate",
                "sample_rate",
                "chunk_ms",
                "split_seconds",
            }
        }
        handshake["authorization"] = {"accessToken": f"Bearer {self.api_key}"}
        final, completed_turns, metadata, errors = "", {}, {}, []
        requested_session_id = str(uuid.uuid4())

        async with connect(
            f"wss://api.meta.ai/v1/asr/realtime?sessionId={requested_session_id}",
            open_timeout=20,
            close_timeout=20,
            ping_interval=None,
            max_size=None,
        ) as websocket:
            await websocket.send(json.dumps(handshake))
            acknowledgement = json.loads(await asyncio.wait_for(websocket.recv(), 20))
            if acknowledgement.get("type") == "error":
                raise RuntimeError(acknowledgement.get("message", "handshake failed"))
            metadata["session_id"] = acknowledgement.get("sessionId", requested_session_id)

            async def receive() -> None:
                nonlocal final
                async for raw in websocket:
                    message = json.loads(raw)
                    kind = message.get("type")
                    if kind == "transcript" and message.get("final"):
                        final = message.get("transcript", "")
                    elif kind == "speechComplete":
                        completed_turns[message["turnId"]] = message.get("transcript", "")
                    elif kind == "audioProgress":
                        metadata["audio_processed_ms"] = message.get("audioProcessedMs")
                    elif kind == "error":
                        errors.append(message.get("message", "unknown server error"))

            receiver = asyncio.create_task(receive())
            try:
                chunk_bytes = sample_rate * 2 * config["chunk_ms"] // 1000
                async for chunk in realtime_chunks(audio, sample_rate, chunk_bytes):
                    await websocket.send(chunk)
                await websocket.send(json.dumps({"type": "endStream"}))
                await asyncio.wait_for(receiver, 120)
            finally:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

        if errors:
            raise RuntimeError(errors[0])
        if config["mode"] != "PUSH_TO_TALK":
            final = " ".join(completed_turns[key].strip() for key in sorted(completed_turns)).strip()
            metadata["completed_turns"] = len(completed_turns)
        return {
            "text": final,
            **metadata,
            "empty_transcript": not bool(final),
            "streamed_pcm_sha256": hashlib.sha256(audio).hexdigest(),
            "streamed_audio_seconds": len(audio) / (sample_rate * 2),
        }


def main() -> None:
    args = parser(
        "Stream the benchmark through Meta Muse Transcribe",
        "meta_muse_transcribe.jsonl",
        concurrency=8,
        starts=1000 / 60,
    ).parse_args()
    run_from_args(MetaMuse, args)


if __name__ == "__main__":
    main()
