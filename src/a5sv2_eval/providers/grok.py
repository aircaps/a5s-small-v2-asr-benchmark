from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from a5sv2_eval.common import parser, realtime_chunks, run_from_args


class Grok:
    system_id = "xai_grok_speech_to_text_streaming"
    system = "xAI Grok Speech-to-Text Streaming"
    model = "grok-speech-to-text-streaming"

    def __init__(self) -> None:
        self.api_key = os.getenv("GROK_API_KEY") or os.environ["XAI_API_KEY"]

    def streaming_config(self, row: dict) -> dict:
        return {
            "sample_rate": row["sample_rate"],
            "encoding": "pcm",
            "interim_results": False,
            "endpointing": 400,
            "language": "en",
            "filler_words": True,
            "chunk_ms": 100,
        }

    def capacity_units(self, row: dict) -> int:
        return 1

    async def transcribe(self, pcm: bytes, config: dict) -> dict:
        query = {
            key: str(value).lower()
            for key, value in config.items()
            if key != "chunk_ms"
        }
        final, segments, metadata, errors = "", [], {"partial_events": 0, "word_count": 0}, []
        completed = asyncio.Event()
        async with connect(
            f"wss://api.x.ai/v1/stt?{urlencode(query)}",
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            open_timeout=20,
            close_timeout=10,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as websocket:
            created = json.loads(await asyncio.wait_for(websocket.recv(), 30))
            if created.get("type") != "transcript.created":
                raise RuntimeError(f"expected transcript.created: {created}")
            metadata["transcript_id"] = created.get("id")

            async def receive() -> None:
                nonlocal final
                async for raw in websocket:
                    message = json.loads(raw)
                    kind = message.get("type")
                    if kind == "transcript.partial":
                        metadata["partial_events"] += 1
                        if message.get("speech_final") and message.get("text", "").strip():
                            segments.append(message["text"].strip())
                            metadata["word_count"] += len(message.get("words", []))
                    elif kind == "transcript.done":
                        tail = message.get("text", "").strip()
                        final = " ".join([*segments, *([tail] if tail else [])])
                        metadata["duration_seconds"] = message.get("duration")
                        metadata["language"] = message.get("language")
                        metadata["finalized_segments"] = len(segments)
                        metadata["word_count"] += len(message.get("words", []))
                        completed.set()
                    elif kind == "error":
                        errors.append(message.get("message") or json.dumps(message))
                        completed.set()

            receiver = asyncio.create_task(receive())
            try:
                chunk_bytes = config["sample_rate"] * 2 * config["chunk_ms"] // 1000
                async for chunk in realtime_chunks(pcm, config["sample_rate"], chunk_bytes):
                    await websocket.send(chunk)
                await websocket.send(json.dumps({"type": "audio.done"}))
                await asyncio.wait_for(completed.wait(), 120)
                if errors:
                    raise RuntimeError(errors[0])
            finally:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

        return {"text": final, **metadata, "empty_transcript": not bool(final.strip())}


def main() -> None:
    args = parser(
        "Stream the benchmark through xAI Grok Speech to Text",
        "xai_grok_speech_to_text_streaming.jsonl",
        concurrency=100,
        starts=600,
    ).parse_args()
    run_from_args(Grok, args)


if __name__ == "__main__":
    main()
