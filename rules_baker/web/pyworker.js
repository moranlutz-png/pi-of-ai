// Python sandbox worker — runs Python in-browser via Pyodide (WASM), offline.
// Two ops:
//   op:'run'   — execute code, capture stdout / errors (used by tests)
//   op:'check' — analyze ONE code block: syntax error? names not defined here?
//                and if clean, execute it and capture output. Returns a report.
importScripts('https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js');

let py = null;
(async () => {
  try { py = await loadPyodide(); self.postMessage({ type: 'ready' }); }
  catch (e) { self.postMessage({ type: 'ready', error: String(e && e.message || e) }); }
})();

// Static analyzer: finds syntax errors and every name used-but-not-defined-here,
// then (if clean) executes and captures output. Returns JSON.
const CHECK_PY = String.raw`
import ast, builtins, json, io, contextlib

def _check(src):
    rep = {"syntax_error": None, "undefined": [], "ran": False, "out": "", "err": None}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        rep["syntax_error"] = f"{e.msg} (line {e.lineno})"
        return rep
    defined = set(dir(builtins)) | {"self", "cls", "__name__", "__file__", "__doc__"}

    class Defs(ast.NodeVisitor):
        def _args(self, a):
            for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
                defined.add(x.arg)
            if a.vararg: defined.add(a.vararg.arg)
            if a.kwarg: defined.add(a.kwarg.arg)
        def visit_FunctionDef(self, n):
            defined.add(n.name); self._args(n.args); self.generic_visit(n)
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_Lambda(self, n):
            self._args(n.args); self.generic_visit(n)
        def visit_ClassDef(self, n):
            defined.add(n.name); self.generic_visit(n)
        def visit_Import(self, n):
            for a in n.names: defined.add((a.asname or a.name).split(".")[0])
            self.generic_visit(n)
        def visit_ImportFrom(self, n):
            for a in n.names: defined.add(a.asname or a.name)
            self.generic_visit(n)
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Store): defined.add(n.id)
            self.generic_visit(n)

    Defs().visit(tree)
    used = []

    class Uses(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Load) and n.id not in defined and n.id not in used:
                used.append(n.id)

    Uses().visit(tree)
    rep["undefined"] = used
    if not used:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(tree, "<code>", "exec"), {})
            rep["ran"] = True
        except Exception as e:
            rep["err"] = f"{type(e).__name__}: {e}"
        rep["out"] = buf.getvalue()
    return rep

json.dumps(_check(SRC))
`;

// Pyodide ships a minimal stdlib; sqlite3, ssl, decimal and friends are
// separate packages that must be fetched before the import works. Without this
// a perfectly correct `import sqlite3` fails with ModuleNotFoundError, and the
// auto-fix loop then burns every attempt on a problem the model cannot fix.
//
// Downloads happen once, then come from the browser cache. A 'progress' message
// tells the page a fetch is underway so its timeout does not count network time
// against the code's execution budget.
async function loadImports(op, code) {
  const before = Object.keys(py.loadedPackages || {});
  try {
    self.postMessage({ type: 'progress', op, phase: 'packages' });
    await py.loadPackagesFromImports(code, { messageCallback: () => {}, errorCallback: () => {} });
  } catch (err) {
    // An unknown import is not fatal — let the code run and report the real
    // error, which is more useful than a loader failure.
    console.warn('[sandbox] package load:', err && err.message || err);
  }
  const after = Object.keys(py.loadedPackages || {});
  return after.filter((p) => !before.includes(p));
}

self.onmessage = async (e) => {
  const { op, code } = e.data;
  if (!py) { self.postMessage({ type: 'result', op, err: 'sandbox not ready' }); return; }

  if (op === 'check') {
    try {
      const loaded = await loadImports(op, code);
      py.globals.set('SRC', code);
      const res = await py.runPythonAsync(CHECK_PY);
      let rep;
      try { rep = JSON.parse(res); } catch (_) { rep = { err: 'analyzer parse failed' }; }
      if (loaded.length) rep.packages = loaded;
      self.postMessage({ type: 'result', op: 'check', rep });
    } catch (err) {
      self.postMessage({ type: 'result', op: 'check', rep: { err: String(err && err.message || err) } });
    }
    return;
  }

  // op === 'run' (execute + capture) — used by tests
  let out = '';
  const cap = { batched: (s) => { out += s + '\n'; } };
  try {
    await loadImports(op, code);
    py.setStdout(cap); py.setStderr(cap);
    await py.runPythonAsync(code);
    self.postMessage({ type: 'result', op: 'run', ok: true, out, err: '' });
  } catch (err) {
    self.postMessage({ type: 'result', op: 'run', ok: false, out, err: String(err && err.message || err) });
  }
};
