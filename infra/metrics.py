#!/usr/bin/env python3

import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timed(operation: str, extra: dict | None = None):
	start = time.time()
	try:
		yield
	finally:
		duration = time.time() - start
		meta = {"op": operation, "duration_s": round(duration, 4)}
		if extra:
			meta.update(extra)
		logger.info(f"metrics {meta}")

