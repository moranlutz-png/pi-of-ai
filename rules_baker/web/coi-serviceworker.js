/* Pi-of-AI — cross-origin isolation shim.
 *
 * WHY THIS EXISTS
 * ---------------
 * Multi-threaded wllama needs SharedArrayBuffer. Browsers only hand that out to
 * a "cross-origin isolated" page, which requires two response headers:
 *
 *     Cross-Origin-Opener-Policy:   same-origin
 *     Cross-Origin-Embedder-Policy: require-corp
 *
 * `web/serve.py` sets them, so locally this file does nothing at all. But static
 * hosts like GitHub Pages can't set headers — and without them the page silently
 * falls back to the single-thread WASM build, which is slowest on exactly the
 * low-end hardware this project targets.
 *
 * A service worker sits between the page and the network, so it can add those
 * headers to responses itself. First visit: register, reload once, and every
 * later request comes back isolated.
 *
 * ONE FILE, TWO CONTEXTS
 * ----------------------
 * This same file runs as the page script AND as the service worker. `window` is
 * undefined inside a worker, which is how it tells the two apart.
 *
 * LIMITS
 * ------
 *  - Needs a secure context (https, or localhost). file:// will not work.
 *  - Cross-origin subresources (the wllama WASM on jsDelivr, GGUF weights on
 *    huggingface.co) are re-served with Cross-Origin-Resource-Policy so that
 *    require-corp doesn't block them.
 */

if (typeof window === 'undefined') {
  // ---------------------------------------------------------------- worker --
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

  self.addEventListener('fetch', (event) => {
    const req = event.request;

    // Cache-only probes must pass straight through; intercepting them throws.
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;

    event.respondWith(
      fetch(req)
        .then((res) => {
          // Opaque responses have no readable body or headers — leave untouched.
          if (res.status === 0) return res;

          const headers = new Headers(res.headers);
          headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
          headers.set('Cross-Origin-Opener-Policy', 'same-origin');
          // Lets require-corp accept the CDN + Hugging Face downloads.
          headers.set('Cross-Origin-Resource-Policy', 'cross-origin');

          return new Response(res.body, {
            status: res.status,
            statusText: res.statusText,
            headers,
          });
        })
        .catch((err) => {
          console.error('[coi] fetch failed:', err);
          throw err;
        })
    );
  });
} else {
  // ------------------------------------------------------------------ page --
  (() => {
    const RELOAD_KEY = 'coi-reload-attempted';

    // Already isolated (e.g. serve.py sent the headers) — nothing to do.
    if (window.crossOriginIsolated) {
      sessionStorage.removeItem(RELOAD_KEY);
      return;
    }

    if (!window.isSecureContext) {
      console.warn('[coi] not a secure context — WASM threads unavailable, using single-thread.');
      return;
    }
    if (!('serviceWorker' in navigator)) {
      console.warn('[coi] no service worker support — using single-thread.');
      return;
    }

    // Guarded so a host that can never isolate doesn't reload forever.
    const reloadOnce = () => {
      if (sessionStorage.getItem(RELOAD_KEY)) {
        console.warn('[coi] still not isolated after a reload — staying single-thread.');
        return;
      }
      sessionStorage.setItem(RELOAD_KEY, '1');
      window.location.reload();
    };

    const src = document.currentScript && document.currentScript.src;
    if (!src) {
      console.warn('[coi] could not determine own URL — skipping registration.');
      return;
    }

    navigator.serviceWorker.register(src).then(
      (registration) => {
        registration.addEventListener('updatefound', reloadOnce);
        // Registered on a previous visit but not yet controlling this page.
        if (registration.active && !navigator.serviceWorker.controller) reloadOnce();
      },
      (err) => console.warn('[coi] registration failed — using single-thread.', err)
    );
  })();
}
