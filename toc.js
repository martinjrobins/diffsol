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
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="primer/modelling_with_diffsol.html"><strong aria-hidden="true">1.</strong> Modelling with Diffsol</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="primer/first_order_odes.html"><strong aria-hidden="true">1.1.</strong> Explicit First Order ODEs</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="primer/population_dynamics.html"><strong aria-hidden="true">1.1.1.</strong> Example: Population Dynamics</a></li></ol></li><li class="chapter-item expanded "><a href="primer/higher_order_odes.html"><strong aria-hidden="true">1.2.</strong> Higher Order ODEs</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="primer/spring_mass_systems.html"><strong aria-hidden="true">1.2.1.</strong> Example: Spring-mass systems</a></li></ol></li><li class="chapter-item expanded "><a href="primer/discrete_events.html"><strong aria-hidden="true">1.3.</strong> Discrete Events</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="primer/compartmental_models_of_drug_delivery.html"><strong aria-hidden="true">1.3.1.</strong> Example: Compartmental models of Drug Delivery</a></li><li class="chapter-item expanded "><a href="primer/bouncing_ball.html"><strong aria-hidden="true">1.3.2.</strong> Example: Bouncing Ball</a></li></ol></li><li class="chapter-item expanded "><a href="primer/the_mass_matrix.html"><strong aria-hidden="true">1.4.</strong> DAEs via the Mass Matrix</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="primer/electrical_circuits.html"><strong aria-hidden="true">1.4.1.</strong> Example: Electrical Circuits</a></li></ol></li><li class="chapter-item expanded "><a href="primer/pdes.html"><strong aria-hidden="true">1.5.</strong> PDEs</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="primer/heat_equation.html"><strong aria-hidden="true">1.5.1.</strong> Example: Heat Equation</a></li><li class="chapter-item expanded "><a href="primer/physics_based_battery_simulation.html"><strong aria-hidden="true">1.5.2.</strong> Example: Physics-based Battery Simulation</a></li></ol></li><li class="chapter-item expanded "><a href="primer/forward_sensitivity_analysis.html"><strong aria-hidden="true">1.6.</strong> Forward Sensitivity Analysis</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="primer/population_dynamics_fitting.html"><strong aria-hidden="true">1.6.1.</strong> Example: Fitting a predator-prey model to data</a></li></ol></li><li class="chapter-item expanded "><a href="primer/backwards_sensitivity_analysis.html"><strong aria-hidden="true">1.7.</strong> Backwards Sensitivity Analysis</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="primer/spring_mass_fitting.html"><strong aria-hidden="true">1.7.1.</strong> Example: Fitting a spring-mass model to data</a></li><li class="chapter-item expanded "><a href="primer/weather_neural_ode.html"><strong aria-hidden="true">1.7.2.</strong> Example: Weather prediction using neural ODEs</a></li></ol></li></ol></li><li class="chapter-item expanded "><a href="specify/specifying_the_problem.html"><strong aria-hidden="true">2.</strong> Diffsol APIs for specifying problems</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="specify/diffsl/diffsl.html"><strong aria-hidden="true">2.1.</strong> DiffSL</a></li><li class="chapter-item expanded "><a href="specify/closure/rust_closures.html"><strong aria-hidden="true">2.2.</strong> Rust closures</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="specify/closure/explicit.html"><strong aria-hidden="true">2.2.1.</strong> Explicit</a></li><li class="chapter-item expanded "><a href="specify/closure/implicit.html"><strong aria-hidden="true">2.2.2.</strong> Implicit</a></li><li class="chapter-item expanded "><a href="specify/closure/mass_matrix.html"><strong aria-hidden="true">2.2.3.</strong> Mass matrix</a></li><li class="chapter-item expanded "><a href="specify/closure/root_finding.html"><strong aria-hidden="true">2.2.4.</strong> Root finding</a></li><li class="chapter-item expanded "><a href="specify/closure/forward_sensitivity.html"><strong aria-hidden="true">2.2.5.</strong> Forward Sensitivity</a></li><li class="chapter-item expanded "><a href="specify/closure/sparse_problems.html"><strong aria-hidden="true">2.2.6.</strong> Sparse problems</a></li></ol></li><li class="chapter-item expanded "><a href="specify/trait/ode_equations_trait.html"><strong aria-hidden="true">2.3.</strong> OdeEquations trait</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="specify/trait/non_linear_functions.html"><strong aria-hidden="true">2.3.1.</strong> Non-linear functions</a></li><li class="chapter-item expanded "><a href="specify/trait/constant_functions.html"><strong aria-hidden="true">2.3.2.</strong> Constant functions</a></li><li class="chapter-item expanded "><a href="specify/trait/linear_functions.html"><strong aria-hidden="true">2.3.3.</strong> Linear functions</a></li><li class="chapter-item expanded "><a href="specify/trait/ode_systems.html"><strong aria-hidden="true">2.3.4.</strong> ODE systems</a></li></ol></li></ol></li><li class="chapter-item expanded "><a href="solver/creating_a_solver.html"><strong aria-hidden="true">3.</strong> Creating a solver</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="solver/initialisation.html"><strong aria-hidden="true">3.1.</strong> Initialisation</a></li><li class="chapter-item expanded "><a href="solver/tableau.html"><strong aria-hidden="true">3.2.</strong> Tableau</a></li></ol></li><li class="chapter-item expanded "><a href="solve/solving_the_problem.html"><strong aria-hidden="true">4.</strong> Solving the problem</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="solve/manual-time-stepping.html"><strong aria-hidden="true">4.1.</strong> Manual time-stepping</a></li><li class="chapter-item expanded "><a href="solve/interpolation.html"><strong aria-hidden="true">4.2.</strong> Interpolation</a></li><li class="chapter-item expanded "><a href="solve/stopping.html"><strong aria-hidden="true">4.3.</strong> Stopping</a></li><li class="chapter-item expanded "><a href="solve/forward_sens.html"><strong aria-hidden="true">4.4.</strong> Forward Sensitivities</a></li></ol></li><li class="chapter-item expanded "><a href="benchmarks/benchmarks.html"><strong aria-hidden="true">5.</strong> Benchmarks</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="benchmarks/sundials.html"><strong aria-hidden="true">5.1.</strong> Sundials</a></li><li class="chapter-item expanded "><a href="benchmarks/python.html"><strong aria-hidden="true">5.2.</strong> Python (Diffrax &amp; Casadi)</a></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
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
