"""Payload-verified RS(255,k) evaluation on the original BPSK symbol axis.

This module generates coded inputs and evaluates receiver outputs. Transmitted
bits and original payloads belong to the evaluator: neither the receiver nor
the decoder may use them to choose an operation. All comparisons must use the
same k (159 by default), input frame and errors-and-erasures decoder. Changing
k is a separate, predeclared experiment, never an adaptive receiver action.

Requires reedsolo==1.7.0. Process workers are supported. Do not share a codec
between concurrent threads: reedsolo maintains module-level Galois-field tables.
There is no CRC and no extra decoder acceptance policy. A wrong returned payload
is counted by evaluator truth comparison, not treated as an online detection.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import numbers
import time

import numpy as np
from reedsolo import RSCodec, ReedSolomonError


RS_N = 255
DEFAULT_K = 159
REEDSOLO_VERSION = "1.7.0"


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return value


@lru_cache(maxsize=8)
def _codec(k):
    """A process-local codec with every coding convention fixed explicitly."""
    if version("reedsolo") != REEDSOLO_VERSION:
        raise RuntimeError(f"Coding protocol requires reedsolo=={REEDSOLO_VERSION}")
    return RSCodec(nsym=RS_N - k, nsize=RS_N, fcr=0, prim=0x11D,
                   generator=2, c_exp=8, single_gen=True)


def _bit_digest(bits):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return hashlib.sha256(len(bits).to_bytes(8, "big") +
                          np.packbits(bits, bitorder="big").tobytes()).hexdigest()


def make_coded_frame(seed, n_symbols, k=DEFAULT_K):
    """Return (bits, payloads, codewords, meta); fill the remainder randomly.

    One BPSK symbol carries one transmitted bit (0 -> -1, 1 -> +1).
    floor(n_symbols / (255*8)) full codewords occupy the start of the frame.
    Any trailing filler belongs to the observation window and full-frame
    goodput denominator, but is never presented to the RS decoder.
    """
    seed = _integer(seed, "seed", 0)
    n_symbols = _integer(n_symbols, "n_symbols", RS_N * 8)
    k = _integer(k, "k", 1, RS_N - 1)
    codec = _codec(k)
    n_words = n_symbols // (RS_N * 8)
    n_coded = n_words * RS_N * 8
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    payloads = rng.integers(0, 256, (n_words, k), dtype=np.uint8)
    codewords = np.stack([
        np.frombuffer(bytes(codec.encode(payload.tobytes())), dtype=np.uint8).copy()
        for payload in payloads
    ])
    coded_bits = np.unpackbits(codewords.reshape(-1), bitorder="big")
    filler = rng.integers(0, 2, n_symbols - n_coded, dtype=np.uint8)
    bits = np.concatenate((coded_bits, filler))
    meta = {
        "n": RS_N, "k": k, "parity_bytes": RS_N - k, "gf_bits": 8,
        "primitive_polynomial": 0x11D, "generator": 2, "first_consecutive_root": 0,
        "bitorder": "big", "bpsk_mapping": "0:-1,1:+1; decision soft>0",
        "n_symbols": n_symbols, "n_codewords": n_words,
        "n_coded_bits": n_coded, "n_payload_bits": n_words * k * 8,
        "n_filler_bits": n_symbols - n_coded,
        "seed": seed, "payload_seed_sequence": [seed, 0],
        "reedsolo_version": REEDSOLO_VERSION,
        "crc": "none", "success_rule": "returned payload equals original payload",
        "erasure_rule": "any erased bit erases its containing byte",
        "tx_bits_sha256": _bit_digest(bits),
        "payloads_sha256": hashlib.sha256(payloads.tobytes()).hexdigest(),
        "codewords_sha256": hashlib.sha256(codewords.tobytes()).hexdigest(),
    }
    return bits, payloads, codewords, meta


def _decode_received(codec, rx_bytes, erased_bytes, parity):
    """Decoder boundary: this function deliberately receives no ground truth."""
    erase_positions = np.flatnonzero(erased_bytes).tolist()
    start = time.perf_counter()
    if len(erase_positions) > parity:
        return {"payload": None, "returned": False,
                "failure_reason": "erasure_budget_exceeded",
                "exception": None, "attempted": False,
                "elapsed": time.perf_counter() - start}
    try:
        decoded = codec.decode(rx_bytes.tobytes(), erase_pos=erase_positions)
    except ReedSolomonError as exc:
        return {"payload": None, "returned": False,
                "failure_reason": "reed_solomon_error",
                "exception": type(exc).__name__, "attempted": True,
                "elapsed": time.perf_counter() - start}
    payload = bytes(decoded[0] if isinstance(decoded, tuple) else decoded)
    return {"payload": payload, "returned": True, "failure_reason": None,
            "exception": None, "attempted": True,
            "elapsed": time.perf_counter() - start}


def _score_word(codec, rx_bytes, erased_bytes, payload, original, k):
    decoded = _decode_received(codec, rx_bytes, erased_bytes, RS_N - k)
    success = bool(decoded["returned"] and decoded["payload"] == payload.tobytes())
    wrong_return = bool(decoded["returned"] and not success)
    byte_error = rx_bytes != original
    errors = int(np.count_nonzero(byte_error & ~erased_bytes))
    erasures = int(np.count_nonzero(erased_bytes))
    return {
        "success": success, "decoder_returned": decoded["returned"],
        "decoder_attempted": decoded["attempted"],
        "decoder_exception": decoded["exception"],
        "decoder_failure_reason": decoded["failure_reason"],
        "undetected_wrong_payload": wrong_return, "wrong_return": wrong_return,
        "erroneous_bytes_total": int(np.count_nonzero(byte_error)),
        "erroneous_bytes_outside_erasures": errors,
        "erased_bytes": erasures, "two_e_plus_v": 2 * errors + erasures,
        "inside_guaranteed_decoding_radius": 2 * errors + erasures <= RS_N - k,
        "raw_payload_wrong": bool(np.any(rx_bytes[:k] != payload)),
        "payload_success_bits": k * 8 if success else 0,
        "decode_elapsed_seconds": float(decoded["elapsed"]),
        "rx_codeword_sha256": hashlib.sha256(rx_bytes.tobytes()).hexdigest(),
        "byte_erasure_mask_sha256": _bit_digest(erased_bytes),
    }


def _validate_truth(truth_bits, payloads, codewords, meta):
    """Reject mismatched evidence or code conventions before scoring."""
    k = _integer(meta.get("k"), "meta.k", 1, RS_N - 1)
    n_symbols = _integer(meta.get("n_symbols"), "meta.n_symbols", RS_N * 8)
    n_words = n_symbols // (RS_N * 8)
    expected = {
        "n": RS_N, "parity_bytes": RS_N - k, "gf_bits": 8,
        "primitive_polynomial": 0x11D, "generator": 2, "first_consecutive_root": 0,
        "bitorder": "big", "n_codewords": n_words,
        "n_coded_bits": n_words * RS_N * 8,
        "n_payload_bits": n_words * k * 8,
        "n_filler_bits": n_symbols - n_words * RS_N * 8,
        "reedsolo_version": REEDSOLO_VERSION, "crc": "none",
    }
    for name, value in expected.items():
        if meta.get(name) != value:
            raise ValueError(f"Coding metadata mismatch: {name}")
    truth_bits = np.asarray(truth_bits)
    payloads = np.asarray(payloads)
    codewords = np.asarray(codewords)
    if truth_bits.shape != (n_symbols,) or not np.isin(truth_bits, (0, 1)).all():
        raise ValueError("truth_bits must contain the complete binary symbol axis")
    if payloads.dtype != np.uint8 or payloads.shape != (n_words, k):
        raise ValueError("Payload shape/dtype differs from metadata")
    if codewords.dtype != np.uint8 or codewords.shape != (n_words, RS_N):
        raise ValueError("Codeword shape/dtype differs from metadata")
    if not np.array_equal(codewords[:, :k], payloads):
        raise ValueError("Payload differs from systematic codeword prefix")
    if not np.array_equal(np.packbits(truth_bits[:expected["n_coded_bits"]],
                                     bitorder="big"), codewords.reshape(-1)):
        raise ValueError("Transmitted coded bits differ from original codewords")
    digests = {"tx_bits_sha256": _bit_digest(truth_bits),
               "payloads_sha256": hashlib.sha256(payloads.tobytes()).hexdigest(),
               "codewords_sha256": hashlib.sha256(codewords.tobytes()).hexdigest()}
    if any(meta.get(name) != digest for name, digest in digests.items()):
        raise ValueError("Original coding evidence digest mismatch")
    return truth_bits.astype(np.uint8, copy=False), payloads, codewords, k


def evaluate_coded(soft, erasure_mask, truth_bits, payloads, codewords, meta):
    """Return JSON-safe decoding counts and goodput, with per-word evidence.

    soft and erasure_mask preserve all original symbol indices, including
    erased symbols and filler. No time-axis compression is permitted. BER is
    reported with the retained-bit denominator; all correctly decoded payload
    bits use explicit full-frame and coded-transmission denominators.
    """
    truth_bits, payloads, codewords, k = _validate_truth(
        truth_bits, payloads, codewords, meta)
    n_symbols = meta["n_symbols"]
    n_words = meta["n_codewords"]
    n_coded = meta["n_coded_bits"]
    soft = np.asarray(soft)
    mask_input = np.asarray(erasure_mask)
    if soft.shape != (n_symbols,) or mask_input.shape != (n_symbols,):
        raise ValueError("Receiver output/mask must preserve the complete symbol axis")
    if not np.isrealobj(soft) or not np.all(np.isfinite(soft)):
        raise ValueError("Receiver soft output must be finite and real")
    if not np.isin(mask_input, (0, 1)).all():
        raise ValueError("Erasure mask must be Boolean or contain only 0/1")
    erasure_mask = mask_input.astype(bool, copy=False)
    hard_bits = (soft > 0).astype(np.uint8)
    rx_bytes = np.packbits(hard_bits[:n_coded], bitorder="big").reshape(n_words, RS_N)
    byte_mask = erasure_mask[:n_coded].reshape(n_words, RS_N, 8).any(axis=2)
    codec = _codec(k)
    words = []
    for index in range(n_words):
        word = _score_word(codec, rx_bytes[index], byte_mask[index],
                           payloads[index], codewords[index], k)
        word["codeword_index"] = index
        words.append(word)
    successes = sum(int(word["success"]) for word in words)
    wrong_returns = sum(int(word["wrong_return"]) for word in words)
    decoder_failures = sum(int(not word["decoder_returned"]) for word in words)
    payload_bits = successes * k * 8
    coded_keep = ~erasure_mask[:n_coded]
    retained_count = int(np.count_nonzero(coded_keep))
    retained_errors = int(np.count_nonzero(
        (hard_bits[:n_coded] != truth_bits[:n_coded]) & coded_keep))
    return {
        "codewords": words, "n_codewords": n_words,
        "n_codeword_successes": successes, "n_codeword_failures": n_words - successes,
        "n_decoder_failures": decoder_failures, "n_wrong_returns": wrong_returns,
        "n_undetected_wrong_payloads": wrong_returns,
        "n_decoder_exceptions": sum(w["decoder_exception"] is not None for w in words),
        "n_decoder_attempts": sum(int(w["decoder_attempted"]) for w in words),
        "n_over_budget_erasure_words": sum(
            w["decoder_failure_reason"] == "erasure_budget_exceeded" for w in words),
        "n_raw_wrong_payloads": sum(int(w["raw_payload_wrong"]) for w in words),
        "bler": (n_words - successes) / n_words,
        "frame_all_codewords_success": successes == n_words,
        "n_payload_success_bits": payload_bits,
        "n_payload_bits_offered": meta["n_payload_bits"],
        "n_transmitted_coded_bits": n_coded, "n_total_frame_bits": n_symbols,
        "n_filler_bits": meta["n_filler_bits"],
        "net_goodput_per_coded_transmitted_bit": payload_bits / n_coded,
        "useful_decoded_bits_per_full_frame_symbol": payload_bits / n_symbols,
        "payload_delivery_fraction": payload_bits / meta["n_payload_bits"],
        "code_rate": k / RS_N, "n_parity_bits": n_words * (RS_N - k) * 8,
        "coding_overhead_fraction_of_coded_bits": (RS_N - k) / RS_N,
        "n_erroneous_bytes_outside_erasures": sum(
            w["erroneous_bytes_outside_erasures"] for w in words),
        "n_erased_bytes": int(byte_mask.sum()),
        "n_erased_coded_bits": int(erasure_mask[:n_coded].sum()),
        "n_erased_full_frame_bits": int(erasure_mask.sum()),
        "n_retained_coded_bits": retained_count,
        "n_retained_coded_bit_errors": retained_errors,
        "retained_coded_ber": retained_errors / retained_count if retained_count else None,
        "coded_bit_retention": retained_count / n_coded,
        "full_bit_retention": float(np.mean(~erasure_mask)),
        "decode_elapsed_seconds": sum(w["decode_elapsed_seconds"] for w in words),
        "symbol_erasure_mask_sha256": _bit_digest(erasure_mask),
        "coded_byte_erasure_mask_sha256": _bit_digest(byte_mask),
        "hard_bits_sha256": _bit_digest(hard_bits),
    }


def self_test():
    """Exercise all error/erasure boundary partitions and evaluator semantics."""
    bits, payloads, codewords, meta = make_coded_frame(92841, 10000)
    repeated = make_coded_frame(92841, 10000)
    assert np.array_equal(bits, repeated[0]) and meta == repeated[3]
    soft = 2.0 * bits.astype(float) - 1.0
    clean = evaluate_coded(soft, np.zeros(len(bits), bool), bits, payloads, codewords, meta)
    assert clean["n_codeword_successes"] == 4 and clean["bler"] == 0
    assert clean["n_payload_success_bits"] == 4 * DEFAULT_K * 8
    assert clean["net_goodput_per_coded_transmitted_bit"] == DEFAULT_K / RS_N
    assert clean["useful_decoded_bits_per_full_frame_symbol"] == 0.5088
    codec = _codec(DEFAULT_K)
    parity = RS_N - DEFAULT_K
    rng = np.random.default_rng(834244)
    checks = 0
    # Every possible number of unknown byte errors at 2e+v = 96 is tested.
    # Changing data at erased positions prevents a trivially clean input.
    for n_errors in range(parity // 2 + 1):
        for inside_offset in (0, 1):
            n_erased = parity - 2 * n_errors - inside_offset
            if n_erased < 0:
                continue
            positions = rng.permutation(RS_N)
            corrupt = codewords[0].copy()
            changed = positions[:n_errors + n_erased]
            corrupt[changed] ^= rng.integers(1, 256, len(changed), dtype=np.uint8)
            mask = np.zeros(RS_N, bool)
            mask[positions[:n_erased]] = True
            result = _score_word(codec, corrupt, mask, payloads[0], codewords[0], DEFAULT_K)
            assert result["success"] and result["two_e_plus_v"] == parity - inside_offset
            checks += 1
    excessive = _score_word(codec, codewords[0], np.arange(RS_N) < parity + 1,
                            payloads[0], codewords[0], DEFAULT_K)
    assert not excessive["success"] and not excessive["decoder_attempted"]
    assert excessive["decoder_failure_reason"] == "erasure_budget_exceeded"
    # A valid different codeword produces no decoder error; truth accounting
    # must reject it as successfully delivered data, even without a CRC.
    other = payloads[0].copy()
    other[0] ^= np.uint8(1)
    other_word = np.frombuffer(bytes(codec.encode(other.tobytes())), dtype=np.uint8)
    wrong_soft = soft.copy()
    wrong_soft[:RS_N * 8] = 2.0 * np.unpackbits(other_word).astype(float) - 1.0
    wrong = evaluate_coded(wrong_soft, np.zeros(len(bits), bool), bits,
                           payloads, codewords, meta)
    assert wrong["n_wrong_returns"] == 1 and wrong["n_decoder_failures"] == 0
    assert wrong["n_codeword_failures"] == 1 and wrong["n_codeword_successes"] == 3
    assert wrong["n_payload_success_bits"] == 3 * DEFAULT_K * 8
    mask = np.zeros(len(bits), bool)
    mask[[0, 7, 8, meta["n_coded_bits"]]] = True
    mapped = evaluate_coded(soft, mask, bits, payloads, codewords, meta)
    assert mapped["n_erased_bytes"] == 2 and mapped["n_erased_coded_bits"] == 3
    assert mapped["n_erased_full_frame_bits"] == 4 and mapped["bler"] == 0
    all_erased = evaluate_coded(soft, np.ones(len(bits), bool), bits,
                                payloads, codewords, meta)
    assert all_erased["retained_coded_ber"] is None
    assert all_erased["n_decoder_attempts"] == 0 and all_erased["bler"] == 1
    # Generalize frame length and optional k without changing the default.
    for n_symbols, k in ((2040, 159), (4097, 127)):
        b, p, c, m = make_coded_frame(152, n_symbols, k)
        r = evaluate_coded(2.0 * b.astype(float) - 1.0, np.zeros(n_symbols, bool), b, p, c, m)
        assert r["bler"] == 0 and r["n_codewords"] == n_symbols // 2040
        assert r["n_filler_bits"] == n_symbols % 2040
    for invalid_call in (
            lambda: make_coded_frame(3, 2039),
            lambda: make_coded_frame(3, 10000, 255),
            lambda: evaluate_coded(soft[:-1], mask[:-1], bits, payloads, codewords, meta),
            lambda: evaluate_coded(soft, mask, bits, payloads, codewords,
                                    dict(meta, n_coded_bits=1))):
        try:
            invalid_call()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid coding input was accepted")
    json.dumps(clean, allow_nan=False)
    json.dumps(all_erased, allow_nan=False)
    return {"self_test": "passed", "reedsolo_version": REEDSOLO_VERSION,
            "boundary_and_near_boundary_cases": checks,
            "checks": ["deterministic payload generation", "goodput denominators",
                       "all errors/erasures boundary partitions",
                       "over-budget erasures skipped", "wrong valid payload accounting",
                       "any-bit erasure mapping and filler exclusion",
                       "general frame lengths and optional k=127",
                       "compressed axis and metadata mismatch rejected",
                       "JSON-safe empty retained-bit denominator"]}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, allow_nan=False))
