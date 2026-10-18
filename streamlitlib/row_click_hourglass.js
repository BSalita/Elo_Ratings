function armRankingRowHourglass(doc) {
  if (!doc || !doc.documentElement) return;
  var css = "html.ranking-row-wait, html.ranking-row-wait * { cursor: wait !important; }";
  doc.documentElement.classList.add("ranking-row-wait");
  var style = doc.getElementById("ranking-row-wait-style");
  if (!style) {
    style = doc.createElement("style");
    style.id = "ranking-row-wait-style";
    (doc.head || doc.documentElement).appendChild(style);
  }
  style.textContent = css;
}

function clearRankingRowHourglassWhenGridsPaint(doc) {
  if (!doc || !doc.documentElement) return;
  if (!doc.documentElement.classList.contains("ranking-row-wait")) return;
  var started = Date.now();

  function clear() {
    doc.documentElement.classList.remove("ranking-row-wait");
    var style = doc.getElementById("ranking-row-wait-style");
    if (style) style.remove();
  }

  function frameLabel(frame) {
    return (frame.getAttribute("src") || "") + " " + (frame.getAttribute("title") || "");
  }

  function isGridFrame(frame) {
    return /aggrid/i.test(frameLabel(frame));
  }

  function hasGridRoot(frame) {
    try {
      return !!(frame.contentDocument && frame.contentDocument.querySelector(".ag-root"));
    } catch (err) {
      return false;
    }
  }

  function tick() {
    var frames = Array.prototype.filter.call(doc.querySelectorAll("iframe"), isGridFrame);
    var pending = 0;
    var ready = 0;
    for (var i = 0; i < frames.length; i += 1) {
      if (hasGridRoot(frames[i])) ready += 1;
      else pending += 1;
    }
    var elapsed = Date.now() - started;
    if (pending === 0 && (ready > 0 || elapsed > 1000)) {
      clear();
      return;
    }
    if (elapsed > 20000) {
      clear();
      return;
    }
    setTimeout(tick, 50);
  }

  tick();
}
