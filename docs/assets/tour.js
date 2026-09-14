// Mount the asciinema player on anything carrying a cast URL.
//
// Not a plain DOMContentLoaded listener: `navigation.instant` is on, so moving
// between pages swaps the DOM without a page load and that event never fires
// again. Material publishes every such swap on `document$`, which also emits
// once for the initial load, so subscribing to it covers both; the listener is
// the fallback for a build with instant navigation turned off.
(function () {
  function mount() {
    document.querySelectorAll("[data-cast]").forEach(function (el) {
      if (el.dataset.mounted) return;
      el.dataset.mounted = "1";
      AsciinemaPlayer.create(el.dataset.cast, el, {
        fit: "width",
        // Chapters, from the marker events scripts/tour.py writes.
        markers: true,
        // A frame with a run in it, rather than the empty list it opens on.
        poster: el.dataset.poster || "npt:0:45",
        terminalFontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
      });
    });
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(mount);
  } else {
    document.addEventListener("DOMContentLoaded", mount);
  }
})();
