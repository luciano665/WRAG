import re

WH_WORDS = r"(?:what|how|why|when|where|who|which|should|do|does|did|is|are|can|could|would|will|has|have|had)"
SEP_WORDS = r"(?:and|or)"
OBJ_PREPS = r"(?:of|in|for|to|about|regarding|with|between|on|at|over|under|into|from)"

def _clean_piece(s: str) -> str:
  s = s.strip().strip(" .,!?:;")
  return s

def _clean_question(s: str) -> str:
  s = _clean_piece(s)
  if not s:
    return ""
  # Capitalize first character (keep acronyms intact later if needed)
  s = s[0].upper() + s[1:]
  if not s.endswith("?"):
    s += "?"
  # Normalize double punctuation that might slip in
  s = re.sub(r"\?{2,}$", "?", s)
  return s

def _split_items(tail: str) -> list[str]:
  # Split by commas and "and/or" while ignoring empty parts
  parts = re.split(r"\s*,\s*|\s+(?:and|or)\s+", tail, flags=re.IGNORECASE)
  return [p for p in ( _clean_piece(x) for x in parts ) if p]

def _has_wh(s: str) -> bool:
  return re.search(rf"\b{WH_WORDS}\b", s, flags=re.IGNORECASE) is not None

def _try_year_vs_year(q: str):
  # e.g., "How did X change in 2020 vs 2021?" -> two questions with years swapped in place
  m = re.search(r"^(.*?)(\b(19|20)\d{2})\s+vs\s+((19|20)\d{2})(.*)$", q.strip(), flags=re.IGNORECASE)
  if not m:
    return None
  pre, y1, _, y2, _, post = m.groups()
  left = _clean_question(f"{pre}{y1}{post}")
  right = _clean_question(f"{pre}{y2}{post}")
  return [left, right]

def _try_general_vs(q: str):
  # Generic "X vs Y" -> two questions; if no WH stem, fall back to pros/cons framing
  if " vs " not in q.lower():
    return None
  parts = re.split(r"\s+vs\s+", q.strip(), flags=re.IGNORECASE)
  if len(parts) != 2:
    return None
  left, right = _clean_piece(parts[0]), _clean_piece(parts[1])
  # If the left side already looks like a full question (has WH or starts with modal/aux), mirror the stem
  if _has_wh(q) or re.match(rf"^\b(should|do|does|did|is|are|can|could|would|will|has|have|had)\b", q.strip(), flags=re.IGNORECASE):
    return [_clean_question(left), _clean_question(right)]
  # Otherwise, use a neutral comparison prompt
  return [
    _clean_question(f"What are the pros and cons of {left}"),
    _clean_question(f"What are the pros and cons of {right}")
  ]

def _try_multi_wh_split(q: str):
  # Split on "and/or" ONLY when followed by a new WH/aux start (avoids breaking object lists)
  pattern = rf"\s+(?:{SEP_WORDS})\s+(?=(?:{WH_WORDS})\b)"
  parts = re.split(pattern, q.strip(), flags=re.IGNORECASE)
  parts = [p for p in ( _clean_piece(x) for x in parts ) if p]
  if len(parts) > 1:
    return [_clean_question(p) for p in parts]
  return None

def _try_prefix_object_list(q: str):
  """
  Detect a stem like:
    'List the causes of ...'
    'Compare the economic growth of ...'
    'Should companies invest in ...'
    'How does AI affect ...'
  and then split the tail object list: 'X, Y and Z'
  Only applies if the tail has no WH-words (to avoid mixing with multi-wh questions).
  """
  # Capture a reasonable stem that ends with a preposition expecting an object
  m = re.match(rf"^(.*?\b{OBJ_PREPS}\s+)(.+)$", q.strip(), flags=re.IGNORECASE)
  if not m:
    return None
  prefix, tail = m.groups()
  if _has_wh(tail):
    return None  # let multi-wh handle those cases
  items = _split_items(tail)
  if len(items) < 2:
    return None
  return [_clean_question(prefix + it) for it in items]

def _try_compare_a_and_b(q: str):
  # Fallback for "Compare A and B" without a prepositional stem
  m = re.match(r"^\s*compare\s+(.+?)\s+(?:and|or)\s+(.+?)\s*\??$", q, flags=re.IGNORECASE)
  if not m:
    return None
  a, b = _clean_piece(m.group(1)), _clean_piece(m.group(2))
  return [_clean_question(f"Compare {a}"), _clean_question(f"Compare {b}")]

def decompose(question: str) -> list[str]:
  q = question.strip()

  # 1) Special: YEAR vs YEAR
  out = _try_year_vs_year(q)
  if out:
    return out

  # 2) General "X vs Y"
  out = _try_general_vs(q)
  if out:
    return out

  # 3) Multi-WH splits like "... and how/what/why ..."
  out = _try_multi_wh_split(q)
  if out:
    return out

  # 4) Prefix + object list (handles lists and "A and/or B" after a preposition)
  out = _try_prefix_object_list(q)
  if out:
    return out

  # 5) "Compare A and B" without explicit preposition
  out = _try_compare_a_and_b(q)
  if out:
    return out

  # 6) Last-resort: split on "and/or" if it looks like two peers and not just a phrase
  #    (keeps the left stem for the right fragment)
  if re.search(rf"\b{SEP_WORDS}\b", q, flags=re.IGNORECASE):
    # Try to infer a stem boundary; use everything before the last "and/or" as stem
    parts = re.split(rf"\b{SEP_WORDS}\b", q, flags=re.IGNORECASE)
    parts = [p for p in ( _clean_piece(x) for x in parts ) if p]
    if len(parts) > 1:
      stem = _clean_piece(parts[0])
      return [_clean_question(stem)] + [_clean_question(f"{stem} {p}") for p in parts[1:]]

  # 7) Default: single question cleaned
  return [_clean_question(q)]

###-----TEST CASES-----

print(decompose("What are the benefits of AI and what are the risks?"))
# -> ['What are the benefits of AI?', 'What are the risks?']

print(decompose("Should companies invest in AI or blockchain?"))
# -> ['Should companies invest in AI?', 'Should companies invest in blockchain?']

print(decompose("AI vs human decision making"))
# -> ['What are the pros/cons of AI?', 'What are the pros/cons of human decision making?']

print(decompose("List the causes of inflation, unemployment, and poverty."))
# -> ['List the causes of inflation?', 'List the causes of unemployment?', 'List the causes of poverty?']

print(decompose("How does AI affect jobs and how does it affect education?"))
# -> ['How does AI affect jobs?', 'How does it affect education?']

print(decompose("Compare the economic growth of China and India."))
# -> ['What is the economic growth of China?', 'What is the economic growth of India?']

print(decompose("How did climate policy change in 2020 vs 2021?"))
# -> ['How did climate policy change in 2020?', 
#     'How did climate policy change in 2021?']
