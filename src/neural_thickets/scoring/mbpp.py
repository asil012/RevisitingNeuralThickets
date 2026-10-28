from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import signal
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace

from neural_thickets.scoring.sandbox import isolated_call, isolation_available
from neural_thickets.scoring.types import Outcome


def extract_mbpp(response: str) -> str:
    for pattern in (
        r"<answer>(.*?)</answer>",
        r"```python\n?(.*?)```",
        r"```\n?(.*?)```",
    ):
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()
    return ""


def score_mbpp_isolated(
    response: str,
    tests: Sequence[str],
    setup: str = "",
    *,
    timeout_seconds: float = 30,
    memory_gib: float = 8,
    unsafe_allow_unisolated: bool = False,
) -> Outcome:
    """Run model-generated code in a disposable subprocess.

    This preserves the reference all-tests-pass metric while preventing a bad
    generation from hanging or exhausting the inference process.
    """

    code = extract_mbpp(response)
    if not code:
        return _outcome(code, False, "empty", "no code extracted")
    available, _ = isolation_available()
    if available:
        result = isolated_call(
            lambda: _execute(code, tuple(tests), setup, timeout_seconds=5),
            timeout_seconds=int(timeout_seconds),
        )
        if result["ok"]:
            correct, status, reason = result["result"]
            return _outcome(code, bool(correct), status, reason)
        return _outcome(code, False, result["status"], result["reason"])
    if not unsafe_allow_unisolated:
        raise RuntimeError(
            "Secure MBPP isolation is unavailable. Run in the documented "
            "capability-limited Linux container, or pass unsafe_allow_unisolated=True "
            "only inside a disposable VM."
        )
    context = mp.get_context("spawn")
    recv, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send, code, tuple(tests), setup, memory_gib))
    process.start()
    send.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return _outcome(code, False, "timeout", f"wall timeout after {timeout_seconds}s")
    result = recv.recv() if recv.poll() else (False, "process_failure", "worker returned no result")
    recv.close()
    correct, status, reason = result
    return _outcome(code, bool(correct), status, reason)


def score_mbpp_batch_isolated(
    cases: Sequence[tuple[str, Sequence[str], str]],
    *,
    row_timeout_seconds: int = 5,
    memory_gib: float = 8,
    unsafe_allow_unisolated: bool = False,
) -> tuple[Outcome, ...]:
    """Score a shard in one disposable process, with a per-row alarm."""

    if not cases:
        raise ValueError("MBPP scoring requires at least one case")
    available, reason = isolation_available()
    if not available and not unsafe_allow_unisolated:
        raise RuntimeError(
            f"{reason}. MBPP scoring fails closed; use a disposable, capability-limited Linux "
            "container or explicitly opt into unsafe unisolated execution inside a VM."
        )
    normalized = tuple((response, tuple(tests), setup) for response, tests, setup in cases)
    context = mp.get_context("spawn")
    descriptor, result_path = tempfile.mkstemp(prefix="neural_thickets_mbpp_batch_", suffix=".json")
    os.close(descriptor)
    process = context.Process(
        target=_batch_worker,
        args=(result_path, normalized, row_timeout_seconds, memory_gib, available),
    )
    process.start()
    process.join(max(60, len(cases) * (row_timeout_seconds + 1)))
    if process.is_alive():
        process.terminate()
        process.join()
        results = tuple(
            _fallback_score(
                response,
                tests,
                setup,
                row_timeout_seconds,
                memory_gib,
                unsafe_allow_unisolated,
            )
            for response, tests, setup in normalized
        )
        os.unlink(result_path)
        return results
    try:
        with open(result_path, encoding="utf-8") as handle:
            payload = handle.read()
    finally:
        os.unlink(result_path)
    try:
        results = json.loads(payload) if payload else []
    except json.JSONDecodeError:
        results = []
    if not results:
        results = [
            (False, "process_failure", "batch worker returned no valid result")
        ] * len(cases)
    if len(results) != len(cases):
        raise RuntimeError(
            f"MBPP batch returned {len(results)} results for {len(cases)} cases"
        )
    return tuple(
        _outcome(extract_mbpp(response), bool(correct), status, reason)
        for (response, _, _), (correct, status, reason) in zip(normalized, results)
    )


def _fallback_score(
    response: str,
    tests: tuple[str, ...],
    setup: str,
    timeout_seconds: int,
    memory_gib: float,
    unsafe_allow_unisolated: bool,
) -> Outcome:
    outcome = score_mbpp_isolated(
        response,
        tests,
        setup,
        timeout_seconds=timeout_seconds + 2,
        memory_gib=memory_gib,
        unsafe_allow_unisolated=unsafe_allow_unisolated,
    )
    return replace(outcome, scoring_status=f"fallback_{outcome.scoring_status}")


def _worker(send: object, code: str, tests: tuple[str, ...], setup: str, memory_gib: float) -> None:
    _limit_memory(memory_gib)
    send.send(_execute(code, tests, setup, timeout_seconds=5))
    send.close()


def _batch_worker(
    result_path: str,
    cases: tuple[tuple[str, tuple[str, ...], str], ...],
    timeout_seconds: int,
    memory_gib: float,
    secure: bool,
) -> None:
    _limit_memory(memory_gib)
    passed = []
    for response, tests, setup in cases:
        code = extract_mbpp(response)
        if secure:
            result = isolated_call(
                lambda code=code, tests=tests, setup=setup: _execute(
                    code,
                    tests,
                    setup,
                    timeout_seconds,
                ),
                timeout_seconds=timeout_seconds + 2,
            )
            if result["ok"]:
                passed.append(result["result"])
            else:
                passed.append((False, result["status"], result["reason"]))
        else:
            passed.append(_execute(code, tests, setup, timeout_seconds))
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(passed, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _execute(
    code: str,
    tests: tuple[str, ...],
    setup: str,
    timeout_seconds: int,
) -> tuple[bool, str, str | None]:
    if not code:
        return False, "empty", "no code extracted"
    namespace: dict[str, object] = {}
    try:
        with _alarm(timeout_seconds):
            if setup.strip():
                exec(setup, namespace)
            exec(code, namespace)
        for test in tests:
            with _alarm(timeout_seconds):
                exec(test, namespace)
        return True, "passed", None
    except AssertionError as error:
        return False, "failed", f"AssertionError: {str(error)[:500]}"
    except TimeoutError:
        return False, "timeout", f"per-execution timeout after {timeout_seconds}s"
    except BaseException as error:
        return False, "exception", f"{type(error).__name__}: {str(error)[:500]}"


@contextmanager
def _alarm(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def timeout_handler(signum: int, frame: object) -> None:
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _limit_memory(memory_gib: float) -> None:
    if memory_gib <= 0:
        return
    try:
        import resource

        limit = int(memory_gib * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError):
        pass


def _outcome(
    answer: str,
    correct: bool,
    status: str,
    reason: str | None,
) -> Outcome:
    return Outcome(
        answer=answer,
        valid=bool(answer),
        correct=correct,
        reward=float(correct),
        answer_reward=float(correct),
        format_reward=None,
        scoring_status=status,
        scoring_reason=reason,
    )

