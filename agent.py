import os
import time
import signal
import ast
import copy
import re
import json
import threading
import math
import collections
import heapq
import bisect
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Any, Generator, Tuple, Optional

import dotenv
dotenv.load_dotenv()

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions


class KeyManager:
    def __init__(self):
        self.keys = [os.environ.get(f'GEMINI_KEY_{i}') for i in range(1, 6)]
        self.keys = [k for k in self.keys if k]

        if not self.keys:
            single = os.environ.get('GEMINI_API_KEY')
            if single:
                self.keys.append(single)

        if not self.keys:
            raise ValueError("No API keys found")

        self.current_index = 0
        self.configure_current()

    def configure_current(self):
        genai.configure(api_key=self.keys[self.current_index])

    def rotate(self):
        self.current_index = (self.current_index + 1) % len(self.keys)
        self.configure_current()


key_manager = KeyManager()
model = genai.GenerativeModel('gemini-2.5-flash')


@dataclass
class IterationResult:
    iteration: int
    code: str
    correctness: float
    avg_time_ms: float
    complexity: str
    reward: float
    logs: str
    status: str
    test_breakdown: List[str] = field(default_factory=list)
    stream_delta: str = ""


class TimeoutException(Exception):
    pass


class Agent:
    def __init__(self, problem_statement: str):
        self.problem = problem_statement
        self.test_cases: List[Dict[str, Any]] = []
        self.current_code: Optional[str] = None
        self.history: List[IterationResult] = []
        self.max_iterations = 5
        self.timeout_seconds = 2

    def _timeout_handler(self, signum, frame):
        raise TimeoutException()

    def _calculate_reward(self, correctness: float, time_ms: float, complexity: str,
                          passes_all: bool, has_error: bool) -> float:
        reward = correctness * 40

        complexity_score = {
            'O(1)': 25, 'O(log n)': 22, 'O(n)': 18,
            'O(n log n)': 12, 'O(n²)': -5, 'O(n³)': -15, 'O(2^n)': -30
        }
        reward += complexity_score.get(complexity, 0)

        if passes_all:
            if time_ms < 0.05:
                reward += 15
            elif time_ms < 0.5:
                reward += 10
            elif time_ms < 5.0:
                reward += 5
            elif time_ms > 100.0:
                reward -= 5

        if passes_all and correctness == 1.0:
            reward += 20

        if has_error:
            reward -= 20

        return max(0.0, min(100.0, reward))

    def _strip_code_noise(self, source: str) -> str:
        try:
            parsed = ast.parse(source)
            for node in ast.walk(parsed):
                if not isinstance(node, (ast.FunctionDef, ast.ClassDef,
                                         ast.AsyncFunctionDef, ast.Module)):
                    continue
                if not node.body:
                    continue
                if isinstance(node.body[0], ast.Expr):
                    val = getattr(node.body[0], 'value', None)
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        node.body.pop(0)

            if hasattr(ast, 'unparse'):
                return ast.unparse(parsed)
            return source

        except Exception:
            source = re.sub(r'#.*', '', source)
            source = re.sub(r'""".*?"""', '', source, flags=re.DOTALL)
            return '\n'.join(line for line in source.splitlines() if line.strip())

    def _find_solution_function(self, namespace: Dict[str, Any]) -> Any:
        if 'Solution' in namespace:
            try:
                inst = namespace['Solution']()
                for attr in dir(inst):
                    fn = getattr(inst, attr)
                    if callable(fn) and not attr.startswith('_'):
                        return fn
            except Exception:
                pass

        candidates = [
            obj for name, obj in namespace.items()
            if callable(obj)
            and not name.startswith('_')
            and name != 'Solution'
            and getattr(obj, '__module__', None) is None
        ]
        return candidates[-1] if candidates else None

    def _extract_code_block(self, text: str) -> str:
        pattern = r"```(?:python)?\s*(.*?)```"
        m = re.search(pattern, text, re.DOTALL)
        raw = m.group(1).strip() if m else text.strip()
        raw = re.sub(r"^Here is .*?:\s*", "", raw, flags=re.IGNORECASE)
        return self._strip_code_noise(raw)

    def estimate_complexity(self, code: str) -> str:
        prompt = f"Analyze code. Return ONLY Big-O. CODE: {code}"
        for _ in range(len(key_manager.keys) + 1):
            try:
                res = model.generate_content(prompt)
                return res.text.strip()
            except google_exceptions.ResourceExhausted:
                key_manager.rotate()
                time.sleep(1)
            except Exception:
                return "O(?)"
        return "O(?)"

    def _generate_and_stream(self, prompt: str, iteration: int) -> Generator[IterationResult, None, str]:
        full = ""
        retries = len(key_manager.keys) + 1

        for _ in range(retries):
            try:
                stream = model.generate_content(prompt, stream=True)
                for chunk in stream:
                    if chunk.text:
                        full += chunk.text
                        yield IterationResult(iteration, full, 0, 0, "",
                                              0, "", "streaming", stream_delta=chunk.text)
                return full
            except google_exceptions.ResourceExhausted:
                key_manager.rotate()
                full = ""
                time.sleep(1)
            except Exception as e:
                yield IterationResult(iteration, full, 0, 0, "",
                                      0, f"API Error: {e}", "error")
                return full
        return full

    def generate_self_tests(self) -> Generator[IterationResult, None, None]:
        yield IterationResult(0, "", 0, 0, "", 0,
                              "Analyzing problem & generating test cases...", "starting")

        prompt = f"""
        You are a QA Engineer. Generate 4 robust test cases for this problem.
        Problem: {self.problem}

        Output JSON only:
        {{
            "cases": [
                {{"input": [2,3], "expected": 5}},
                {{"input": [[1,2],[3,4]], "expected": [4,6]}}
            ]
        }}
        """

        text = ""
        for _ in range(len(key_manager.keys) + 1):
            try:
                r = model.generate_content(prompt)
                text = r.text.strip()
                break
            except google_exceptions.ResourceExhausted:
                key_manager.rotate()
                time.sleep(1)
            except Exception as e:
                yield IterationResult(0, "", 0, 0, "",
                                      0, f"Test Gen Error: {e}", "error")
                self.test_cases = [{'input': [], 'expected': None}]
                return

        try:
            m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
            raw = m.group(1) if m else text
            data = json.loads(raw)
            self.test_cases = data['cases']
            yield IterationResult(0, "", 0, 0, "",
                                  0, f"Generated {len(self.test_cases)} tests.",
                                  "tests_generated")
        except Exception:
            self.test_cases = [{'input': [], 'expected': None}]

    def run_tests(self, code: str) -> Tuple[float, float, bool, str, List[str]]:
        try:
            import typing

            ns = {
                'List': typing.List, 'Dict': typing.Dict, 'Tuple': typing.Tuple,
                'Optional': typing.Optional, 'Any': typing.Any, 'Set': typing.Set,
                'math': math, 'collections': collections, 'heapq': heapq,
                'bisect': bisect, 'itertools': itertools, 're': re
            }

            code = code.replace("```python", "").replace("```", "").strip()
            exec(code, ns)
            func = self._find_solution_function(ns)

            if not func:
                return 0.0, 0.0, False, "No callable function found", []

            passed = 0
            total_ms = 0
            errors = []
            breakdown = []

            use_signals = hasattr(signal, 'SIGALRM') and threading.current_thread() is threading.main_thread()
            if use_signals:
                signal.signal(signal.SIGALRM, self._timeout_handler)

            for i, test in enumerate(self.test_cases):
                try:
                    args = copy.deepcopy(test['input'])
                    if use_signals:
                        signal.alarm(self.timeout_seconds)

                    start = time.perf_counter()
                    res = func(*args) if isinstance(args, (list, tuple)) else func(args)
                    end = time.perf_counter()

                    if use_signals:
                        signal.alarm(0)

                    dur = (end - start) * 1000
                    total_ms += dur

                    if res == test['expected']:
                        passed += 1
                        breakdown.append(f"Test {i+1}:Passed ({dur:.4f}ms)")
                    else:
                        msg = f"Test {i+1}:Expected {test['expected']}, got {res}"
                        errors.append(msg)
                        breakdown.append(msg)
                except TimeoutException:
                    msg = f"Test {i+1}:TLE(>{self.timeout_seconds}s)"
                    errors.append(msg)
                    breakdown.append(msg)
                except Exception as e:
                    if use_signals:
                        signal.alarm(0)
                    msg = f"Test {i+1}:Runtime Error: {e}"
                    errors.append(msg)
                    breakdown.append(msg)

            count = len(self.test_cases)
            correctness = passed / count if count else 0
            avg = total_ms / count if count else 0
            all_passed = passed == count

            summary = "All tests passed successfully." if all_passed else "\n".join(errors[:3])
            if len(errors) > 3:
                summary += f"\n...and {len(errors)-3} more errors."

            return correctness, avg, all_passed, summary, breakdown

        except SyntaxError as e:
            return 0.0, 0.0, False, f"Syntax Error: {e.msg} on line {e.lineno}", []
        except Exception as e:
            return 0.0, 0.0, False, f"Critical Sandbox Error: {e}", []

    def solve_generator(self) -> Generator[IterationResult, None, None]:
        yield from self.generate_self_tests()

        if not self.test_cases or self.test_cases == [{'input': [], 'expected': None}]:
            yield IterationResult(0, "", 0, 0, "", 0,
                                  "Could not generate tests. Aborting.", "error")
            return

        yield IterationResult(0, "", 0, 0, "", 0,
                              "Generating initial solution...", "starting")

        prompt_init = f"""
        You are a Python Algorithm Engine.
        Write a Python function for this problem.
        Problem: {self.problem}
        Test Context: {json.dumps(self.test_cases[:1])}
        Do not use comments or markdown. Return raw Python code only.
        """

        response = yield from self._generate_and_stream(prompt_init, 0)
        self.current_code = self._extract_code_block(response)

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            correctness, avg, all_passed, logs, breakdown = self.run_tests(self.current_code)
            complexity = self.estimate_complexity(self.current_code)
            has_error = "Error" in logs or "Exceeded" in logs
            reward = self._calculate_reward(correctness, avg, complexity, all_passed, has_error)

            result = IterationResult(
                iteration, self.current_code, correctness,
                avg, complexity, reward, logs, "complete", breakdown
            )
            self.history.append(result)
            yield result

            if all_passed and reward >= 95:
                yield IterationResult(iteration, self.current_code, 1.0,
                                      avg, complexity, reward,
                                      "Perfect solution found!", "finished")
                break

            if iteration < self.max_iterations:
                yield IterationResult(iteration, "", 0, 0, "",
                                      0, f"Refining Code (Iter {iteration})...", "improving")

                prompt_refine = f"""
                Optimize this solution.
                Problem: {self.problem}
                Current Code: {self.current_code}
                Stats: Accuracy {correctness*100}%, Time {avg:.4f}ms, Complexity {complexity}
                Logs: {logs}
                Return raw Python code only, no markdown.
                """

                response = yield from self._generate_and_stream(prompt_refine, iteration)
                new_code = self._extract_code_block(response)

                if len(new_code) > 10:
                    self.current_code = new_code
                else:
                    yield IterationResult(iteration, "", 0, 0, "",
                                          0, "Model failed to generate valid code. Retrying.", "warning")
