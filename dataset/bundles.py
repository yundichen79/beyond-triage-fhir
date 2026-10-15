"""Stream entries from large FHIR JSON bundles using the standard JSON decoder."""

import json
from pathlib import Path


def iter_bundle_entries(path):
    """Validate the complete top-level Bundle while retaining one entry at a time."""
    decoder = json.JSONDecoder()
    with Path(path).open() as handle:
        buffer, position, eof = "", 0, False

        def refill():
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            chunk = handle.read(65536)
            buffer += chunk
            eof = not chunk

        def peek():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    return ""
                refill()

        def consume(expected):
            nonlocal position
            if peek() != expected:
                raise ValueError(f"Expected {expected!r} in Bundle {path}")
            position += 1

        def value():
            nonlocal position
            peek()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer, position)
                    position = end
                    return result
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"Malformed Bundle JSON: {path}") from exc
                    if len(buffer) - position > 32 * 1024 * 1024:
                        raise ValueError(f"Single Bundle value exceeds 32 MiB: {path}")
                    refill()

        consume("{")
        resource_type, seen_entry, fields = None, False, set()
        while peek() != "}":
            key = value()
            if not isinstance(key, str) or key in fields:
                raise ValueError(f"Invalid or duplicate Bundle field in {path}")
            fields.add(key)
            consume(":")
            if key == "entry":
                seen_entry = True
                consume("[")
                while peek() != "]":
                    entry = value()
                    if not isinstance(entry, dict):
                        raise ValueError(f"Bundle entry is not an object: {path}")
                    yield entry
                    if peek() != "]":
                        consume(",")
                        if peek() == "]":
                            raise ValueError(f"Trailing comma in Bundle entries: {path}")
                consume("]")
            else:
                decoded = value()
                if key == "resourceType":
                    resource_type = decoded
            if peek() != "}":
                consume(",")
                if peek() == "}":
                    raise ValueError(f"Trailing comma in Bundle object: {path}")
        consume("}")
        if peek() or resource_type != "Bundle" or not seen_entry:
            raise ValueError(f"Not a complete FHIR Bundle: {path}")
