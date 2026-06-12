// MathJax configuration for Material for MkDocs + pymdownx.arithmatex (generic).
// arithmatex emits math wrapped in <... class="arithmatex"> as \(...\) / \[...\],
// so we match that class and re-typeset on navigation (instant loading safe).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
