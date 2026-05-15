"""Speech and audio ML manifest."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import max_value, spread, summary

RUN_TAG = "speech_audio"

DEFAULT_THRESHOLDS = {
    "wer_warning": 0.20,
    "snr_gap_warning": 0.10,
    "speaker_gap_warning": 0.10,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _wer_high(path: Path) -> bool:
    return max_value(path, "wer", "cer") > THRESHOLDS["wer_warning"]


def _wer_summary(path: Path) -> dict:
    return summary(path, "max_error_rate", max_value(path, "wer", "cer"))


def _noise_gap(path: Path) -> bool:
    return spread(path, "wer", "accuracy", "f1") > THRESHOLDS["snr_gap_warning"]


def _noise_summary(path: Path) -> dict:
    return summary(path, "noise_condition_gap", spread(path, "wer", "accuracy", "f1"))


def _speaker_gap(path: Path) -> bool:
    return spread(path, "wer", "accuracy", "f1") > THRESHOLDS["speaker_gap_warning"]


def _speaker_summary(path: Path) -> dict:
    return summary(path, "speaker_group_gap", spread(path, "wer", "accuracy", "f1"))


MANIFEST = [
    {
        "id": "high_transcription_error",
        "title": "Speech transcription error rate is high",
        "evidence": "speech_error_rates.csv",
        "evidence_check": _wer_high,
        "evidence_summary": _wer_summary,
        "severity": "medium",
        "queries": ["automatic speech recognition word error rate domain adaptation", "speech recognition WER robustness fine tuning"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report WER/CER by slice and compare domain adaptation or language-model rescoring.",
        "why_it_matters": "Aggregate ASR quality hides specific failure modes.",
        "next_checks": ["Inspect high-WER utterances.", "Report WER by domain."],
        "success_criteria": ["WER falls below 0.20 or improves over baseline."],
        "references": ["ASR", "word error rate", "domain adaptation"],
    },
    {
        "id": "noise_robustness_gap",
        "title": "Audio performance changes sharply by noise/SNR condition",
        "evidence": "speech_noise.csv",
        "evidence_check": _noise_gap,
        "evidence_summary": _noise_summary,
        "severity": "medium",
        "queries": ["speech recognition noise robustness data augmentation", "audio classification noise robustness benchmark"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add noise augmentation and report metrics by SNR or acoustic condition.",
        "why_it_matters": "Audio systems often degrade outside clean recording conditions.",
        "next_checks": ["Bucket metrics by SNR.", "Compare noise augmentation."],
        "success_criteria": ["Noise-condition gap falls below 0.10."],
        "references": ["noise robustness", "SpecAugment", "audio augmentation"],
    },
    {
        "id": "speaker_group_gap",
        "title": "Speech/audio performance differs across speaker groups",
        "evidence": "speech_speakers.csv",
        "evidence_check": _speaker_gap,
        "evidence_summary": _speaker_summary,
        "severity": "high",
        "queries": ["speech recognition demographic bias speaker accent fairness", "ASR performance disparity accents speaker groups"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report speaker/accent subgroup metrics and rebalance data or adaptation sets.",
        "why_it_matters": "ASR and audio models can fail unevenly by accent, device or speaker group.",
        "next_checks": ["Report WER by speaker group.", "Inspect worst-group samples."],
        "success_criteria": ["Speaker-group gap falls below 0.10 or is mitigated."],
        "references": ["ASR fairness", "accent bias", "speaker adaptation"],
    },
    {
        "id": "streaming_latency_missing",
        "title": "Streaming latency evaluation is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["streaming speech recognition latency evaluation", "real time audio machine learning latency benchmark"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Track real-time factor and p95 latency if the model runs in streaming mode.",
        "why_it_matters": "Audio quality is incomplete without latency for interactive systems.",
        "next_checks": ["Measure real-time factor.", "Track p95 streaming latency."],
        "success_criteria": ["Latency fits the application budget."],
        "references": ["streaming ASR", "real-time factor", "latency"],
    },
    {
        "id": "out_of_domain_audio_missing",
        "title": "Out-of-domain audio evaluation is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["out of domain audio classification robustness", "speech domain shift robustness evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add out-of-domain acoustic validation before deployment.",
        "why_it_matters": "Microphone, room and codec changes can dominate audio model errors.",
        "next_checks": ["Evaluate by device/codec.", "Add external audio set."],
        "success_criteria": ["OOD audio performance is reported and acceptable."],
        "references": ["domain shift", "audio robustness", "OOD evaluation"],
    },
]

