"""Client for the native, non-streaming API used by Vision CLV1."""

import json
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

VISION_URL = "https://api.lab.vispark.in/model/text/vision"
INPUT_RATE = Decimal("7.37")
OUTPUT_RATE = Decimal("22.11")


class VisionError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    learning_usage: int = 0
    units_consumed: Decimal = Decimal(0)
    responses: int = 0

    @property
    def estimated_usd(self):
        return (self.input_tokens * INPUT_RATE + self.output_tokens * OUTPUT_RATE) / 1_000_000

    def as_dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "continual_learning_usage": self.learning_usage,
            "units_consumed": str(self.units_consumed),
            "responses": self.responses,
            "estimated_token_cost_usd": str(self.estimated_usd),
        }


@dataclass(frozen=True)
class Reply:
    text: str
    calls: list
    weights: Optional[str]


class Vision:
    def __init__(self, key, *, timeout=300, opener=None):
        self.key = key
        self.timeout = timeout
        self.opener = opener or build_opener(NoRedirect())
        self.usage = Usage()

    def call(self, *, system, content, tools=(), weights=None):
        if not 1 <= len(content) <= 50:
            raise VisionError("Vision requires 1–50 content items per request")
        payload = {
            "size": "large",
            "stream": False,
            "continual_learning": True,
            "system_message": system,
            "content": content,
        }
        if weights is not None:
            payload["weights"] = weights
        if tools:
            payload["tools"] = list(tools)
        request = Request(
            VISION_URL,
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            {"Content-Type": "application/json", "X-API-Key": self.key},
        )
        return self._parse(self._post(request))

    def _post(self, request):
        for attempt in range(4):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as error:
                status = error.code
                retry_after = error.headers.get("Retry-After", "")
                error.close()
                if status == 429 and attempt < 3:
                    delay = int(retry_after) if retry_after.isdigit() else 2 ** attempt
                    if delay <= 60:
                        time.sleep(delay)
                        continue
                raise VisionError(f"Vision returned HTTP {status}") from None
            except (URLError, OSError):
                raise VisionError("Vision transport failed. Billing for this request is unknown.") from None
            except (ValueError, UnicodeError):
                raise VisionError("Vision returned invalid JSON") from None

    def _parse(self, body):
        if not isinstance(body, dict) or body.get("status") not in ("success", True):
            raise VisionError("Vision did not return a successful response")
        data = body.get("data")
        if not isinstance(data, dict):
            raise VisionError("Vision response has no data object")

        input_tokens = _count(data.get("input_tokens"), "input_tokens")
        output_tokens = _count(data.get("output_tokens"), "output_tokens")
        learning_usage = _count(data.get("continual_learning_usage", 0), "continual_learning_usage")
        units = body.get("units_consumed", data.get("units_consumed", 0))
        if type(units) not in (int, float) or not Decimal(str(units)).is_finite() or units < 0:
            raise VisionError("Vision response has invalid units_consumed")
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.learning_usage += learning_usage
        self.usage.units_consumed += Decimal(str(units))
        self.usage.responses += 1

        weights = data.get("weights")
        if weights is not None and (not isinstance(weights, str) or not weights.strip()):
            raise VisionError("Vision returned invalid learning state")
        if data.get("type") == "text" and isinstance(data.get("content"), str):
            return Reply(data["content"], [], weights)
        calls = data.get("tool_calls")
        if data.get("type") == "tool_calls" and isinstance(calls, list) and calls:
            for call in calls:
                if (not isinstance(call, dict) or not isinstance(call.get("name"), str)
                        or not isinstance(call.get("arguments"), dict)):
                    raise VisionError("Vision returned a malformed tool call")
            return Reply("", calls, weights)
        raise VisionError("Vision returned an unsupported response")


def _count(value, name):
    if type(value) is not int or value < 0:
        raise VisionError(f"Vision response has invalid {name}")
    return value
