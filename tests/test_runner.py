import io
import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import benchmark
from runner import RunError, Task, run, text
from vision import Vision, VisionError


class MockHTTP:
    """Exercise request serialization and reply parsing with no external calls."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append((request, json.loads(request.data)))
        return io.BytesIO(json.dumps(next(self.replies)).encode())


def reply(weights=None, calls=None):
    data = {"input_tokens": 200, "output_tokens": 10, "continual_learning_usage": 5}
    if weights is not None:
        data["weights"] = weights
    if calls is None:
        data.update(type="text", content="Completed")
    else:
        data.update(type="tool_calls", tool_calls=calls)
    return {"status": "success", "data": data, "units_consumed": 0.02}


class Fixture:
    name = "Synthetic AP workflow"
    instructions = "Use the company procedure shown in the available records."
    tools = [
        {"type": "function", "function": {
            "name": "post_bill",
            "description": "Post the current bill with the approved amount.",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "number"}},
                "required": ["amount"],
            },
        }},
    ]

    def __init__(self, count):
        self.count = count
        self.completed = 0
        self.calls = []

    def next_task(self):
        if self.completed == self.count:
            return None
        number = self.completed + 1
        feedback = "Check freight terms before posting." if number == 2 else ""
        return Task(f"bill-{number}", [text(f"Process invoice {number}.")], feedback)

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"posted": True}

    def finish_task(self, task_id, final_text):
        self.completed += 1
        return self.completed != 2


def create_fixture():
    return Fixture(100)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.report = Path(directory.name) / "report.json"

    def test_native_tool_loop_and_learning_across_tasks(self):
        http = MockHTTP([
            reply("learning-from-first-read", [
                {"name": "nonexistent", "arguments": {}},
                {"name": "post_bill", "arguments": {"amount": 90.25}},
            ]),
            reply("learning-after-post"),
            reply("learning-after-second-invoice"),
        ])
        fixture = Fixture(2)
        result = run(fixture, Vision("local-test", opener=http), self.report, count=2, fixture=True)

        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result, json.loads(self.report.read_text()))
        self.assertEqual(result["usage"]["estimated_token_cost_usd"], "0.0050853")
        self.assertEqual(result["usage"]["units_consumed"], "0.06")
        self.assertEqual(fixture.calls, [("post_bill", {"amount": 90.25})])

        (request, first), (_, second), (_, third) = http.requests
        self.assertEqual(request.get_header("X-api-key"), "local-test")
        self.assertTrue(all(payload["size"] == "large" and payload["continual_learning"] is True
                            and payload["stream"] is False for _, payload in http.requests))
        self.assertNotIn("weights", first)
        self.assertEqual(second["weights"], "learning-from-first-read")
        self.assertIn('"posted": true', second["content"][-1]["content"])
        self.assertIn("Unknown tool", second["content"][-1]["content"])
        self.assertEqual(third["weights"], "learning-after-post")
        self.assertIn("Check freight terms", third["content"][0]["content"])
        self.assertNotIn("Process invoice 1", json.dumps(third["content"]))
        self.assertNotIn("learning-after-post", self.report.read_text())

    def test_missing_weights_stops_without_committing(self):
        fixture = Fixture(1)
        with self.assertRaisesRegex(VisionError, "No learning state"):
            run(fixture, Vision("local-test", opener=MockHTTP([reply()])), self.report, count=1)
        self.assertEqual(fixture.completed, 0)
        result = json.loads(self.report.read_text())
        self.assertEqual(result["status"], "interrupted")
        self.assertIsNone(result["success_rate"])
        self.assertEqual(result["usage"]["responses"], 1)

    def test_budget_stops_before_another_billable_request(self):
        http = MockHTTP([reply("first")])
        with self.assertRaisesRegex(RunError, "budget"):
            run(Fixture(2), Vision("local-test", opener=http), self.report,
                count=2, max_usd=Decimal("0.001"))
        self.assertEqual(len(http.requests), 1)
        self.assertEqual(json.loads(self.report.read_text())["completed_tasks"], 1)

    def test_cli_completes_100_mocked_invoices(self):
        http = MockHTTP(reply(f"state-{number}") for number in range(100))
        env = self.report.with_name(".env")
        env.write_text('VISION_API_KEY="local-test"\n')

        def mock_vision(key):
            return Vision(key, opener=http)

        with patch.dict(os.environ, {"VISION_API_KEY": ""}), \
                patch.object(benchmark, "Path", side_effect=lambda path: env if path == ".env" else Path(path)), \
                patch.object(benchmark, "Vision", side_effect=mock_vision), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(benchmark.main([
                "--environment", f"{__name__}:create_fixture",
                "--report", str(self.report),
            ]), 0)
        self.assertEqual(http.requests[0][0].get_header("X-api-key"), "local-test")
        result = json.loads(self.report.read_text())
        self.assertEqual(result["completed_tasks"], 100)
        self.assertEqual(result["passed_tasks"], 99)
        self.assertEqual(result["success_rate"], 0.99)
        self.assertEqual(http.requests[-1][1]["weights"], "state-98")


if __name__ == "__main__":
    unittest.main()
