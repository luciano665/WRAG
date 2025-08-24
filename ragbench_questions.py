from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

# Auto install hook if not already installed
def _ensure_datsets_installed(auto_install: bool):
    try:
        