"""
common.py -- shared helpers for the VibeThinker-3B / OJBench CLR pipeline.

Nothing in here needs a GPU. Scripts 2 and 4 use only this file.
"""
import hashlib
import json
import os

# stage 2 forks subprocesses after the tokenizer has been used; without this
# HF prints a fork warning for every single child
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# JSONL IO
# --------------------------------------------------------------------------

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{i+1} is not valid JSON ({e}); skipping", file=sys.stderr)
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def task_key(row):
    """OJBench has one row per (problem, language). This is the join key used everywhere."""
    return f"{row['id']}::{row['language']}"


# --------------------------------------------------------------------------
# Code extraction
# --------------------------------------------------------------------------

FENCE_RE = re.compile(r"```([A-Za-z0-9_+#.\-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.S)
PY_TAGS = {"python", "python3", "py", "pypy", "pypy3"}
CPP_TAGS = {"cpp", "c++", "cc", "cxx", "c", "cpp17"}


def extract_code(text, language):
    """Return the code the model intended as its final answer, or None.

    Mirrors what an OJ-style extractor does: prefer the LAST fenced block whose
    tag matches the requested language, else the last fenced block of any kind.
    """
    if not text:
        return None
    blocks = FENCE_RE.findall(text)
    if not blocks:
        return None
    want = PY_TAGS if language == "python" else CPP_TAGS
    tagged = [code for tag, code in blocks if tag.lower() in want]
    pool = tagged if tagged else [code for _, code in blocks]
    code = pool[-1].strip("\n")
    return code if code.strip() else None


def code_looks_valid(code, language):
    """Is this a plausibly complete program, or a fragment salvaged from a
    truncated fence? "Has a code block" massively overstates yield when most
    rollouts are cut off mid-answer."""
    if not code or not code.strip():
        return False
    if language == "python":
        try:
            compile(code, "main.py", "exec")
            return True
        except (SyntaxError, ValueError):
            return False
    if "main" not in code:
        return False
    for o, c in (("{", "}"), ("(", ")")):
        if code.count(o) != code.count(c):
            return False
    return True


def code_hash(code):
    """Hash that ignores whitespace-only differences, so trivial duplicates collapse."""
    squeezed = re.sub(r"\s+", " ", code or "").strip()
    return hashlib.sha1(squeezed.encode("utf-8")).hexdigest()[:16]


def fence_for(language):
    return "python" if language == "python" else "cpp"


def wrap_as_content(code, language):
    """The `content` string handed to OJBench. A single clean fenced block."""
    return f"```{fence_for(language)}\n{code}\n```"


# --------------------------------------------------------------------------
# Public sample tests, parsed out of the problem statement
# --------------------------------------------------------------------------
# NOTE: these are the samples printed in the statement, i.e. what a human
# contestant sees before submitting. We never touch OJBench's hidden testdata.

_IN_WORDS = r"(?:样例\s*输入|输入\s*样例|樣例輸入|輸入樣例|Sample\s*Input|Input\s*Sample|Example\s*Input|Sample\s*In)"
_OUT_WORDS = r"(?:样例\s*输出|输出\s*样例|樣例輸出|輸出樣例|Sample\s*Output|Output\s*Sample|Example\s*Output|Sample\s*Out)"
_HDR_RE = re.compile(
    r"(?P<kind>" + _IN_WORDS + r"|" + _OUT_WORDS + r")"
    r"(?:[ \t#*\]】:：\-]*(?P<num>\d+))?"      # a sample number, but only on the same line
    r"[ \t#*\]】:：]*",
    re.I,
)

# lines that mean the sample payload has ended
_STOP_RE = re.compile(
    r"^\s*(?:#{1,6}\s|\*\*\S|[【\[](?:输入|输出|说明|数据|提示|样例|限制)"
    r"|(?:Note|Notes|Explanation|Constraints?|Hints?|Scoring|Subtask|Limits|Remarks)\b"
    r"|(?:请|注意|说明|提示|数据范围))",
    re.I,
)


def _block_after(text, start):
    """Grab the payload following a sample header. Returns (payload, was_fenced)."""
    rest = text[start:]
    m = re.match(r"\s*```[A-Za-z0-9_+#.\-]*[ \t]*\r?\n(.*?)```", rest, re.S)
    if m:
        return m.group(1), True

    nl = rest.find("\n")                      # drop the remainder of the header line
    rest = rest[nl + 1:] if nl != -1 else ""
    nxt = _HDR_RE.search(rest)
    chunk = rest[: nxt.start()] if nxt else rest

    lines = chunk.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    out = []
    for ln in lines:
        if not ln.strip():
            if out:                           # blank line ends an unfenced sample
                break
            continue
        if _STOP_RE.match(ln):
            break
        out.append(ln.rstrip())
    return "\n".join(out), False


# Every OJBench prompt ends with an answer template that contains a ```python
# skeleton. Sample parsers will happily pair a real sample input with that
# skeleton and call it the expected output, which then fails every correct
# program. Cut the statement before it.
_FORMAT_CUT = re.compile(r"\n#{1,6}\s*Format\s*:", re.I)

# statements whose samples were replaced with placeholders carry no real tests
_PLACEHOLDER = re.compile(
    r"(<\s*insert[^>]*>|\[\s*Example\s+(Input|Output)\s*\]|<\s*Your code is here\s*>"
    r"|\[\s*your code here\s*\])", re.I)

# an expected output that is source code is a parser failure, never a real sample
_LOOKS_LIKE_CODE = re.compile(
    r"(def\s+main\s*\(|if\s+__name__|#include\s*<|int\s+main\s*\(|public\s+static\s+void)")

_CODE_TAGS = {"python", "python3", "py", "cpp", "c++", "c", "java", "javascript", "js", "rust", "go"}

# some statements put both halves in one fence: "Input:\n...\nOutput:\n..."
_INLINE_IO = re.compile(r"^\s*Input\s*:?\s*\n(?P<in>.*?)\n\s*Output\s*:?\s*\n(?P<out>.*)$",
                        re.S | re.I)


def _statement_body(prompt):
    m = _FORMAT_CUT.search(prompt or "")
    return prompt[: m.start()] if m else (prompt or "")


def _reject(pair):
    """True if this parsed pair cannot possibly be a real sample test."""
    i, o = pair.get("input", ""), pair.get("output", "")
    for side in (i, o):
        if not side.strip():
            return True
        if not re.search(r"[A-Za-z0-9]", side):     # only backticks/brackets/dashes
            return True
        if _PLACEHOLDER.search(side):
            return True
    if _LOOKS_LIKE_CODE.search(o):
        return True
    return False


def _split_inline_io(pair):
    """'Input:\n3\nOutput:\n6' captured as one side -> split it properly."""
    m = _INLINE_IO.match(pair.get("input", "") or "")
    if m and m.group("out").strip():
        return {"input": m.group("in").rstrip() + "\n",
                "output": m.group("out").rstrip() + "\n",
                "fenced": pair.get("fenced", False)}
    return pair


_BARE_IN = re.compile(r"^[ \t]*#{2,6}[ \t]*(?:Input|输入)[ \t]*#?\s*\d*[ \t]*$", re.I | re.M)
_BARE_OUT = re.compile(r"^[ \t]*#{2,6}[ \t]*(?:Output|输出)[ \t]*#?\s*\d*[ \t]*$", re.I | re.M)
_EXAMPLE_HDR = re.compile(r"^[ \t]*#{2,6}[ \t]*(?:Example|Sample|样例|例)s?[ \t]*#?\s*\d*[ \t]*$", re.I | re.M)
_FENCE_BLOCK = re.compile(r"```([A-Za-z0-9_+#.\-]*)[ \t]*\r?\n(.*?)```", re.S)


def _parse_bare_pairs(prompt):
    """`#### Input` / `#### Output` headings. Fenced payloads only -- an unfenced
    one is far more likely to be the *format description* than sample data."""
    hits = [("in", m.end()) for m in _BARE_IN.finditer(prompt)]
    hits += [("out", m.end()) for m in _BARE_OUT.finditer(prompt)]
    hits.sort(key=lambda x: x[1])
    pairs, i = [], 0
    while i < len(hits) - 1:
        if hits[i][0] == "in" and hits[i + 1][0] == "out":
            a, fa = _block_after(prompt, hits[i][1])
            b, fb = _block_after(prompt, hits[i + 1][1])
            if fa and fb and a.strip() and b.strip():
                pairs.append({"input": a.rstrip() + "\n", "output": b.rstrip() + "\n",
                              "fenced": True})
            i += 2
        else:
            i += 1
    return pairs


def _parse_example_section(prompt):
    """An `### Example` heading followed by exactly two fenced blocks."""
    pairs = []
    for m in _EXAMPLE_HDR.finditer(prompt):
        rest = prompt[m.end():]
        nxt = _EXAMPLE_HDR.search(rest)
        chunk = rest[: nxt.start()] if nxt else rest[:4000]
        blocks = [c for tag, c in _FENCE_BLOCK.findall(chunk)
                  if tag.lower() not in _CODE_TAGS and c.strip()]
        if len(blocks) == 1:
            one = _split_inline_io({"input": blocks[0], "fenced": True})
            if "output" in one:
                pairs.append(one)
        elif len(blocks) == 2:
            pairs.append({"input": blocks[0].rstrip() + "\n",
                          "output": blocks[1].rstrip() + "\n", "fenced": True})
    return pairs


def parse_samples(prompt):
    """Return [{'input': str, 'output': str}, ...] parsed from a statement."""
    if not prompt:
        return []
    prompt = _statement_body(prompt)
    # A statement whose samples were stubbed out ("<insert example input here>",
    # "[Example Input]") has no real tests. Those markers also happen to match the
    # sample-header patterns, so bail before parsing rather than filter after.
    if _PLACEHOLDER.search(prompt):
        return []
    hits = []
    for m in _HDR_RE.finditer(prompt):
        kind = "in" if re.match(_IN_WORDS, m.group("kind"), re.I) else "out"
        hits.append((kind, m.end()))
    pairs, i = [], 0
    while i < len(hits) - 1:
        if hits[i][0] == "in" and hits[i + 1][0] == "out":
            inp, f1 = _block_after(prompt, hits[i][1])
            out, f2 = _block_after(prompt, hits[i + 1][1])
            if inp.strip() and out.strip():
                pairs.append({"input": inp.rstrip() + "\n", "output": out.rstrip() + "\n",
                              "fenced": bool(f1 and f2)})
            i += 2
        else:
            i += 1
    if not pairs:
        pairs = _parse_bare_pairs(prompt)
    if not pairs:
        pairs = _parse_example_section(prompt)

    pairs = [_split_inline_io(p) for p in pairs]
    pairs = [p for p in pairs if not _reject(p)]

    # de-duplicate while preserving order
    seen, uniq = set(), []
    for p in pairs:
        sig = (p["input"], p["output"])
        if sig not in seen:
            seen.add(sig)
            uniq.append(p)
    return uniq[:5]


# --------------------------------------------------------------------------
# Output comparison
# --------------------------------------------------------------------------

def norm_output(s):
    if s is None:
        return ""
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


_FLOATISH = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)$")


def outputs_match(got, want, float_tol=1e-6):
    """Token-wise comparison, like a standard OJ checker.

    Exact match on integer-looking tokens; a relative tolerance is allowed only
    when a token is written as a real number, so `0` never matches `7`.
    """
    a, b = norm_output(got), norm_output(want)
    if a == b:
        return True
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y or x.lower() == y.lower():
            continue
        if not (_FLOATISH.match(x) or _FLOATISH.match(y)):
            return False
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > float_tol * max(1.0, abs(fy)):
            return False
    return True


# --------------------------------------------------------------------------
# Sandboxed execution
# --------------------------------------------------------------------------
# We are running model-written code. Do this on a disposable cloud VM only.

def find_pypy3():
    """OJBench judges Python with pypy3, so we verify with pypy3 when available."""
    return shutil.which("pypy3") or shutil.which("pypy")


def _preexec(mem_mb, cpu_s, apply_as):
    def fn():
        os.setsid()
        if apply_as and mem_mb:
            lim = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        if cpu_s:
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s) + 1, int(cpu_s) + 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return fn


def run_program(argv, stdin_text, timeout_s=10.0, mem_mb=1024, cwd=None, apply_as=True):
    """Run argv with stdin_text piped in. Never raises; returns a status dict."""
    try:
        p = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=cwd, text=True, errors="replace",
            preexec_fn=_preexec(mem_mb, timeout_s, apply_as),
        )
    except Exception as e:
        return {"status": "spawn_error", "stdout": "", "stderr": str(e), "time": 0.0}

    t0 = time.time()
    try:
        out, err = p.communicate(input=stdin_text, timeout=timeout_s)
        status = "ok" if p.returncode == 0 else "rte"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        p.communicate()
        return {"status": "tle", "stdout": "", "stderr": "", "time": timeout_s}
    return {"status": status, "stdout": out, "stderr": (err or "")[-1500:],
            "returncode": p.returncode, "time": round(time.time() - t0, 3)}


def compile_cpp(code, workdir, std="c++17"):
    """Compile a C++ source. Returns (exe_path, error_message)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    src, exe = workdir / "main.cpp", workdir / "main"
    src.write_text(code, encoding="utf-8")
    gpp = shutil.which("g++")
    if not gpp:
        return None, "g++ not found on PATH"
    r = subprocess.run(
        [gpp, f"-std={std}", "-O2", "-pipe", "-w", "-o", str(exe), str(src)],
        capture_output=True, text=True, timeout=90,
    )
    if r.returncode != 0:
        return None, (r.stderr or "compile failed")[-1500:]
    return str(exe), None


def run_candidate_on_tests(code, language, tests, workdir, timeout_s=10.0,
                           mem_mb=1024, py_cmd=None, extra_inputs=None):
    """Run one candidate over `tests` (inputs with expected outputs) and over
    `extra_inputs` (generated inputs with no expected output).

    `tests` drive the pass/fail gate. `extra_inputs` exist for differential
    testing: they never affect n_pass, but they do feed the signature, which is
    what clusters candidates by behaviour. On OJBench's NOI half there are no
    statement samples at all, so these inputs are the only clustering signal
    there is -- and the only thing that ever executes a Python candidate.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    extra_inputs = extra_inputs or []
    out = {"compile_error": None, "results": [], "n_pass": 0, "n_total": len(tests),
           "stress_status": [], "n_stress": len(extra_inputs)}

    if language == "cpp":
        exe, err = compile_cpp(code, workdir)
        if err:
            out["compile_error"] = err
            out["signature"] = "CE"
            return out
        argv, apply_as = [exe], True
    else:
        try:                       # free correctness gate when there are no tests
            compile(code, "main.py", "exec")
        except SyntaxError as e:
            out["compile_error"] = f"SyntaxError: {e}"
            out["signature"] = "CE"
            return out
        src = workdir / "main.py"
        src.write_text(code, encoding="utf-8")
        interp = py_cmd or find_pypy3() or sys.executable
        argv = [interp, str(src)]
        # pypy reserves a large virtual arena; an RLIMIT_AS cap makes it die at import
        apply_as = "pypy" not in os.path.basename(interp)

    stdouts = []
    for t in tests:
        r = run_program(argv, t["input"], timeout_s=timeout_s, mem_mb=mem_mb,
                        cwd=str(workdir), apply_as=apply_as)
        ok = r["status"] == "ok" and outputs_match(r["stdout"], t["output"])
        out["results"].append({"status": r["status"], "passed": bool(ok),
                               "time": r.get("time"), "stderr": r.get("stderr", "")[:400],
                               "got": norm_output(r["stdout"])[:2000]})
        stdouts.append(f"{r['status']}\x01{norm_output(r['stdout'])}")
        if ok:
            out["n_pass"] += 1

    for inp in extra_inputs:
        r = run_program(argv, inp, timeout_s=timeout_s, mem_mb=mem_mb,
                        cwd=str(workdir), apply_as=apply_as)
        out["stress_status"].append(r["status"])
        stdouts.append(f"{r['status']}\x01{norm_output(r['stdout'])}")

    out["signature"] = hashlib.sha1("\x02".join(stdouts).encode("utf-8", "replace")).hexdigest()[:16]
    return out


# --------------------------------------------------------------------------
# Tokenizer / chat template
# --------------------------------------------------------------------------
# The model card asks for transformers>=4.54 but 4.45.2 renders VibeThinker's
# template fine in practice. If it ever does not, we fall back to plain ChatML,
# which is what Qwen2.5 finetunes use, or to a file you point VT_CHAT_TEMPLATE at.

CHATML_FALLBACK = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def load_tokenizer(model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    override = os.environ.get("VT_CHAT_TEMPLATE")
    if override:
        tok.chat_template = Path(override).read_text(encoding="utf-8")
        print(f"[tokenizer] chat template overridden from {override}")
    elif getattr(tok, "chat_template", None) is None:
        tok.chat_template = CHATML_FALLBACK
        print("[tokenizer] no chat_template in the repo; using the ChatML fallback. "
              "Check tokenizer_config.json and set VT_CHAT_TEMPLATE if it differs.")
    return tok


def _chatml_render(user_text):
    """Render ChatML by hand. No jinja2, no transformers, no template compilation."""
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def render_chat(tok, user_text):
    if tok is None:
        return _chatml_render(user_text)
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True)
    except ImportError as e:
        # transformers needs jinja2>=3.1.0 to compile a template; some base images ship 3.0.x
        print(f"[tokenizer] {e}")
        print("[tokenizer] falling back to a jinja-free ChatML renderer. "
              "Run `pip install -U 'jinja2>=3.1.4'` to use the model's own template instead.")
        return _chatml_render(user_text)
    except Exception as e:
        print(f"[tokenizer] apply_chat_template failed ({e}); using the ChatML fallback")
        tok.chat_template = CHATML_FALLBACK
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return _chatml_render(user_text)


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

def parse_last_json(text):
    """Pull the last complete top-level JSON object out of a model response."""
    if not text:
        return None
    depth, end, cands = 0, None, []
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "}":
            if depth == 0:
                end = i
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0 and end is not None:
                cands.append(text[i:end + 1])
                end = None
                if len(cands) >= 6:
                    break
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def parse_verdicts(text, m=5):
    """Pull M binary verdicts out of a verification response.

    Tries strict JSON first, then a couple of looser patterns, because a long
    reasoning trace that gets truncated often loses its closing brace but still
    shows the verdict list.
    """
    js = parse_last_json(text)
    if isinstance(js, dict) and isinstance(js.get("verdicts"), list):
        return _norm_verdicts(js["verdicts"], m), str(js.get("worst_issue", ""))[:300]

    hit = re.findall(r'verdicts"?\s*[:=]\s*\[([^\]]*)\]', text or "", re.I)
    if not hit:
        hit = re.findall(r"\[\s*(?:[01]\s*,\s*){%d,}[01]\s*\]" % (m - 2), text or "")
        hit = [h.strip("[]") for h in hit]
    if hit:
        parts = [x.strip().strip('"') for x in hit[-1].split(",")]
        vs = _norm_verdicts(parts, m)
        if vs:
            return vs, ""
    return None, ""


def _norm_verdicts(raw, m):
    vs = [1 if str(x).strip().strip('"').lower() in ("1", "true", "yes") else 0 for x in raw][:m]
    if not vs:
        return None
    vs += [0] * (m - len(vs))
    return vs


def shard_filter(items, shard):
    """`--shard 0/2` keeps every 2nd item starting at 0. For splitting across GPUs."""
    if not shard or shard == "0/1":
        return items
    i, n = (int(x) for x in shard.split("/"))
    return [x for j, x in enumerate(items) if j % n == i]