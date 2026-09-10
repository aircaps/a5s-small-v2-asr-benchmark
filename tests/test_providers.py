import asyncio
import json

import a5sv2_eval.providers.assemblyai as assembly_module
import a5sv2_eval.providers.deepgram as deepgram_module
import a5sv2_eval.providers.elevenlabs as elevenlabs_module
import a5sv2_eval.providers.openai as openai_module
import a5sv2_eval.providers.meta as meta_module
import a5sv2_eval.providers.gemini as gemini_module
import a5sv2_eval.providers.grok as grok_module
from a5sv2_eval.providers.assemblyai import AssemblyAI
from a5sv2_eval.providers.deepgram import Deepgram
from a5sv2_eval.providers.elevenlabs import ElevenLabs
from a5sv2_eval.providers.google import Google, stitch_boundary
from a5sv2_eval.providers.openai import OpenAI
from a5sv2_eval.providers.meta import MetaMuse
from a5sv2_eval.providers.gemini import Gemini
from a5sv2_eval.providers.grok import Grok


class FakeWebSocket:
    def __init__(self, responses):
        self.responses = responses
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    async def recv(self):
        if not self.responses:
            await asyncio.sleep(3600)
        return json.dumps(self.responses.pop(0))

    async def __aiter__(self):
        while self.responses:
            yield json.dumps(self.responses.pop(0))


class FakeConnection:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, *args):
        return False


async def one_chunk(pcm, sample_rate, chunk_bytes):
    yield pcm


def test_provider_configs_preserve_native_rate(monkeypatch):
    for name in [
        "DEEPGRAM_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.setenv(name, "test")
    row = {"sample_rate": 48000}

    assert Deepgram().streaming_config(row)["sample_rate"] == 48000
    assert AssemblyAI().streaming_config(row) == {
        "speech_model": "universal-3-5-pro",
        "mode": "balanced",
        "sample_rate": 48000,
        "encoding": "pcm_s16le",
        "chunk_ms": 100,
    }
    google = object.__new__(Google)
    google.region = "us"
    assert google.streaming_config(row)["sample_rate"] == 48000
    assert google.capacity_units(row) == 6
    assert google.streaming_config(row)["session_audio_seconds"] == 285
    assert ElevenLabs().streaming_config(row)["audio_format"] == "pcm_48000"
    assert OpenAI().streaming_config(row)["sample_rate"] == 24000


def test_google_stitch_boundary_only_removes_exact_normalized_overlap():
    stitched, removed = stitch_boundary(
        "One sentence ends with Hello, WORLD!", "hello world and this continues"
    )
    assert stitched == "One sentence ends with Hello, WORLD! and this continues"
    assert removed == 2

    stitched, removed = stitch_boundary("we really agree", "we agree on this")
    assert stitched == "we really agree we agree on this"
    assert removed == 0

    stitched, removed = stitch_boundary("we heard one two", "one—two more")
    assert stitched == "we heard one two more"
    assert removed == 1


async def test_google_rolls_over_filters_replay_and_records_audit_metadata():
    provider = object.__new__(Google)
    provider.region = "us"
    calls = []
    responses = [
        [{"text": "start shared words", "result_end_seconds": 285.0}],
        [
            {"text": "shared words", "result_end_seconds": 4.0},
            {"text": "shared words finish", "result_end_seconds": 20.0},
        ],
    ]

    async def transcribe_session(pcm, config):
        calls.append(len(pcm))
        return responses[len(calls) - 1]

    provider._transcribe_session = transcribe_session
    config = provider.streaming_config({"sample_rate": 10})
    result = await provider.transcribe(bytes(300 * 10 * 2), config)

    assert calls == [285 * 10 * 2, 20 * 10 * 2]
    assert result["text"] == "start shared words finish"
    assert result["session_count"] == 2
    assert result["streamed_audio_seconds"] == 305
    assert result["segments"][1]["source_start_seconds"] == 280
    assert result["segments"][1]["results_discarded_in_replay"] == 1


async def test_openai_commits_one_manual_turn(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    websocket = FakeWebSocket(
        [
            {"type": "session.updated"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item",
                "transcript": "hello",
                "usage": {"type": "duration", "seconds": 0.1},
            },
        ]
    )
    monkeypatch.setattr(openai_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(openai_module, "realtime_chunks", one_chunk)
    provider = OpenAI()
    result = await provider.transcribe(
        bytes(4800), provider.streaming_config({"sample_rate": 24000})
    )

    assert json.loads(websocket.sent[-1]) == {"type": "input_audio_buffer.commit"}
    assert json.loads(websocket.sent[0])["session"]["audio"]["input"]["transcription"] == {
        "model": "gpt-live-transcribe",
        "languages": ["en"],
        "delay": "xhigh",
    }
    assert result["text"] == "hello"


async def test_deepgram_closes_stream(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    websocket = FakeWebSocket(
        [
            {
                "type": "Results",
                "is_final": True,
                "channel": {"alternatives": [{"transcript": "hello"}]},
                "metadata": {"request_id": "request"},
            },
            {"type": "Metadata", "request_id": "request"},
        ]
    )
    monkeypatch.setattr(deepgram_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(deepgram_module, "realtime_chunks", one_chunk)
    provider = Deepgram()
    result = await provider.transcribe(b"\0\0", provider.streaming_config({"sample_rate": 16000}))

    assert websocket.sent == [b"\0\0", json.dumps({"type": "CloseStream"})]
    assert result["text"] == "hello"


async def test_assemblyai_terminates_stream(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test")
    websocket = FakeWebSocket(
        [
            {"type": "Begin", "id": "session"},
            {"type": "Turn", "end_of_turn": True, "transcript": "hello"},
            {"type": "Termination", "audio_duration_seconds": 1},
        ]
    )
    monkeypatch.setattr(assembly_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(assembly_module, "realtime_chunks", one_chunk)
    provider = AssemblyAI()
    result = await provider.transcribe(b"\0\0", provider.streaming_config({"sample_rate": 16000}))

    assert websocket.sent == [b"\0\0", json.dumps({"type": "Terminate"})]
    assert result["text"] == "hello"


async def test_elevenlabs_commits_last_chunk(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    websocket = FakeWebSocket(
        [
            {"message_type": "session_started", "session_id": "session"},
            {"message_type": "committed_transcript", "text": "hello"},
        ]
    )
    monkeypatch.setattr(elevenlabs_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(elevenlabs_module, "realtime_chunks", one_chunk)
    provider = ElevenLabs()
    result = await provider.transcribe(b"\0\0", provider.streaming_config({"sample_rate": 16000}))

    sent = json.loads(websocket.sent[0])
    assert sent["commit"] is True
    assert sent["sample_rate"] == 16000
    assert result["text"] == "hello"


async def test_meta_handshake_stream_and_end(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test")
    monkeypatch.setenv("META_MODE", "PUSH_TO_TALK")
    websocket = FakeWebSocket(
        [
            {"sessionId": "session"},
            {"type": "audioProgress", "audioProcessedMs": 100},
            {"type": "transcript", "transcript": "hello", "final": True},
        ]
    )
    monkeypatch.setattr(meta_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(meta_module, "realtime_chunks", one_chunk)
    provider = MetaMuse()
    config = provider.streaming_config({"sample_rate": 16000})
    result = await provider.transcribe(b"\0\0", config)

    handshake = json.loads(websocket.sent[0])
    assert handshake["model"] == "muse-voice-transcribe-1.0"
    assert handshake["mode"] == "PUSH_TO_TALK"
    assert handshake["audioEncoding"] == "PCM_16KHZ"
    assert handshake["authorization"] == {"accessToken": "Bearer test"}
    assert websocket.sent[1:] == [b"\0\0", json.dumps({"type": "endStream"})]
    assert result["text"] == "hello"
    assert result["audio_processed_ms"] == 100


async def test_meta_clean_empty_stream_is_an_empty_hypothesis(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test")
    websocket = FakeWebSocket([{"sessionId": "session"}])
    monkeypatch.setattr(meta_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(meta_module, "realtime_chunks", one_chunk)

    result = await MetaMuse().transcribe(
        b"\0\0", MetaMuse().streaming_config({"sample_rate": 16000})
    )

    assert result["text"] == ""
    assert result["empty_transcript"] is True


async def test_meta_endpointing_uses_completed_turns_in_turn_order(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test")
    monkeypatch.setenv("META_MODE", "ENDPOINTING")
    websocket = FakeWebSocket(
        [
            {"sessionId": "session"},
            {"type": "speechComplete", "turnId": 2, "transcript": "world"},
            {"type": "speechComplete", "turnId": 1, "transcript": "hello"},
        ]
    )
    monkeypatch.setattr(meta_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(meta_module, "realtime_chunks", one_chunk)
    provider = MetaMuse()
    config = provider.streaming_config({"sample_rate": 16000})
    result = await provider.transcribe(b"\0\0", config)

    assert result["text"] == "hello world"
    assert provider.system_id == "meta_muse_transcribe"
    assert result["completed_turns"] == 2


async def test_gemini_uses_only_final_transcripts(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("GEMINI_DRAIN_SECONDS", "0.01")
    websocket = FakeWebSocket(
        [
            {"setupComplete": {}},
            {"serverContent": {"interimInputTranscription": {"text": "hel"}}},
            {"serverContent": {"inputTranscription": {"text": "hello"}}},
        ]
    )
    monkeypatch.setattr(gemini_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(gemini_module, "realtime_chunks", one_chunk)
    provider = Gemini()
    result = await provider.transcribe(b"\0\0", provider.streaming_config({"sample_rate": 16000}))

    assert result["text"] == "hello"
    setup = json.loads(websocket.sent[0])["setup"]
    assert setup["inputAudioTranscription"] == {
        "languageCodes": ["en-US"],
        "mode": "VERBATIM",
    }
    assert result["sessions"][0]["server_messages"] == 2
    assert "final_response_delay_seconds" in result["sessions"][0]
    assert json.loads(websocket.sent[-1]) == {"realtimeInput": {"audioStreamEnd": True}}


async def test_grok_keeps_endpointed_segments_and_done_tail(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "test")
    websocket = FakeWebSocket(
        [
            {"type": "transcript.created", "id": "session"},
            {"type": "transcript.partial", "text": "hel", "is_final": False},
            {
                "type": "transcript.partial",
                "text": "hello",
                "is_final": True,
                "speech_final": True,
                "words": [{"text": "hello"}],
            },
            {
                "type": "transcript.done",
                "text": "world",
                "duration": 0.1,
                "language": "en",
                "words": [{"text": "world"}],
            },
        ]
    )
    monkeypatch.setattr(grok_module, "connect", lambda *a, **k: FakeConnection(websocket))
    monkeypatch.setattr(grok_module, "realtime_chunks", one_chunk)
    provider = Grok()
    config = provider.streaming_config({"sample_rate": 16000})
    result = await provider.transcribe(b"\0\0", config)

    assert result["text"] == "hello world"
    assert result["transcript_id"] == "session"
    assert result["word_count"] == 2
    assert websocket.sent == [b"\0\0", json.dumps({"type": "audio.done"})]


async def test_meta_splits_long_sessions_without_dropping_or_replaying_audio(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test")
    provider = MetaMuse()
    replies = iter(["one", "two", "three"])

    async def session(audio, config):
        return {"text": next(replies), "streamed_audio_seconds": len(audio) / 32000}

    monkeypatch.setattr(provider, "_transcribe_session", session)
    config = provider.streaming_config({"sample_rate": 16000}) | {"split_seconds": [0.01, 0.02]}
    result = await provider.transcribe(bytes(800), config)

    assert result["text"] == "one two three"
    assert [(session["source_start_seconds"], session["source_end_seconds"]) for session in result["sessions"]] == [
        (0.0, 0.01),
        (0.01, 0.02),
        (0.02, 0.025),
    ]
