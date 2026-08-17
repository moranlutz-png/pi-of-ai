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

BUILTIN_NAMES = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__builtins__"}

SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _add_target(t, out):
    """Names bound by an assignment target, unpacking tuples and stars."""
    if isinstance(t, ast.Name):
        out.add(t.id)
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            _add_target(e, out)
    elif isinstance(t, ast.Starred):
        _add_target(t.value, out)


def _own_nodes(scope):
    """Nodes belonging to this scope, not descending into nested scopes.

    A nested def still yields the def node itself (it binds a name here), but
    its body belongs to its own scope and is walked separately.
    """
    bodies = []
    if isinstance(scope, ast.Lambda):
        bodies = [scope.body]
    elif isinstance(scope, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        bodies = [scope.elt] + [g.iter for g in scope.generators]
        for g in scope.generators:
            bodies.extend(g.ifs)
    elif isinstance(scope, ast.DictComp):
        bodies = [scope.key, scope.value] + [g.iter for g in scope.generators]
        for g in scope.generators:
            bodies.extend(g.ifs)
    else:
        bodies = list(getattr(scope, "body", []))

    def walk(n):
        yield n
        if isinstance(n, SCOPES):
            return                      # its insides are a different scope
        for c in ast.iter_child_nodes(n):
            yield from walk(c)

    for b in bodies:
        yield from walk(b)


def _bindings(scope):
    """Every name bound directly in this scope."""
    names = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = scope.args
        for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
            names.add(x.arg)
        if a.vararg: names.add(a.vararg.arg)
        if a.kwarg: names.add(a.kwarg.arg)
    if isinstance(scope, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        for g in scope.generators:
            _add_target(g.target, names)

    for n in _own_nodes(scope):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                _add_target(t, names)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            _add_target(n.target, names)
        elif isinstance(n, ast.NamedExpr):
            _add_target(n.target, names)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            _add_target(n.target, names)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars:
                    _add_target(item.optional_vars, names)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                names.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
        elif isinstance(n, ast.MatchAs) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.MatchStar) and n.name:
            names.add(n.name)
    return names


def _nested_scopes(scope):
    for n in _own_nodes(scope):
        if isinstance(n, SCOPES) and n is not scope:
            yield n


def _undefined_names(tree):
    """Names read but never bound anywhere visible from where they are read."""
    found = []

    def visit(scope, visible):
        own = _bindings(scope)
        here = visible | own
        for n in _own_nodes(scope):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id not in here and n.id not in found:
                    found.append(n.id)
        for child in _nested_scopes(scope):
            # A class body's names are NOT visible inside its methods, which is
            # why a method must say self.x rather than x.
            visit(child, visible if isinstance(scope, ast.ClassDef) else here)

    visit(tree, BUILTIN_NAMES)
    return found


def _module_runs_anything(tree):
    """True if the module does more than define things.

    A snippet that only defines functions always 'passes' — nothing it defines
    is ever called, so no bug in it can surface. Worth reporting rather than
    letting a clean tick imply the code was exercised.
    """
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                          ast.Import, ast.ImportFrom)):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue                    # a docstring
        return True
    return False


def _check(src):
    rep = {"syntax_error": None, "undefined": [], "ran": False, "out": "", "err": None,
           "defined_only": False}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        rep["syntax_error"] = f"{e.msg} (line {e.lineno})"
        return rep
    # Scope-aware. The previous version collected every bound name into one
    # global set, so a parameter of one function counted as defined inside
    # another — which let this pass clean:
    #     def a(user_id): return user_id
    #     def b(id):      return user_id      # NameError at runtime
    rep["undefined"] = _undefined_names(tree)
    used = rep["undefined"]
    if not used:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(tree, "<code>", "exec"), {})
            rep["ran"] = True
        except Exception as e:
            rep["err"] = f"{type(e).__name__}: {e}"
        rep["out"] = buf.getvalue()
        # Executing a module that only contains definitions proves nothing about
        # them. Say so rather than letting the tick imply otherwise.
        rep["defined_only"] = rep["ran"] and not _module_runs_anything(tree)
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
