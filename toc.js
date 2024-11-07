// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="specifying_the_problem.html"><strong aria-hidden="true">1.</strong> Specifying the problem</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="ode_equations.html"><strong aria-hidden="true">1.1.</strong> ODE equations</a></li><li class="chapter-item expanded "><a href="mass_matrix.html"><strong aria-hidden="true">1.2.</strong> Mass matrix</a></li><li class="chapter-item expanded "><a href="root_finding.html"><strong aria-hidden="true">1.3.</strong> Root finding</a></li><li class="chapter-item expanded "><a href="forward_sensitivity.html"><strong aria-hidden="true">1.4.</strong> Forward Sensitivity</a></li><li class="chapter-item expanded "><a href="custom_problem_structs.html"><strong aria-hidden="true">1.5.</strong> Custom Problem Structs</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="non_linear_functions.html"><strong aria-hidden="true">1.5.1.</strong> Non-linear functions</a></li><li class="chapter-item expanded "><a href="constant_functions.html"><strong aria-hidden="true">1.5.2.</strong> Constant functions</a></li><li class="chapter-item expanded "><a href="linear_functions.html"><strong aria-hidden="true">1.5.3.</strong> Linear functions</a></li><li class="chapter-item expanded "><a href="putting_it_all_together.html"><strong aria-hidden="true">1.5.4.</strong> Putting it all together</a></li></ol></li><li class="chapter-item expanded "><a href="diffsl.html"><strong aria-hidden="true">1.6.</strong> DiffSL</a></li><li class="chapter-item expanded "><a href="sparse_problems.html"><strong aria-hidden="true">1.7.</strong> Sparse problems</a></li></ol></li><li class="chapter-item expanded "><a href="choosing_a_solver.html"><strong aria-hidden="true">2.</strong> Choosing a solver</a></li><li class="chapter-item expanded "><a href="initialisation.html"><strong aria-hidden="true">3.</strong> Initialisation</a></li><li class="chapter-item expanded "><a href="solving_the_problem.html"><strong aria-hidden="true">4.</strong> Solving the problem</a></li><li class="chapter-item expanded "><a href="benchmarks.html"><strong aria-hidden="true">5.</strong> Benchmarks</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
