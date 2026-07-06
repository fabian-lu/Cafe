import './style.css';

/**
 * CAFE landing page interactions:
 * - terminal typing animation
 * - mobile nav toggle
 * - smooth anchor scrolling offset for fixed nav
 */

const terminalCommands = [
  { text: "cafe run example --smoke", output: "ok: 4 configs × 1 input × 1 rep = 4 runs", delay: 900 },
  { text: "cafe validate my_rag.py", output: "design: full_factorial -> 18 configurations", delay: 800 },
  { text: "python -m study.evaluate", output: "✓ 54/54 cells complete  →  54 judged  →  52 usable verdicts", delay: 1200 },
];

function typeTerminal() {
  const area = document.getElementById("typing-area");
  const output = document.getElementById("terminal-output");
  const chart = document.getElementById("ascii-chart");
  if (!area || !output) return;

  let cmdIndex = 0;
  let charIndex = 0;

  function typeChar() {
    const cmd = terminalCommands[cmdIndex];
    if (charIndex < cmd.text.length) {
      area.textContent += cmd.text.charAt(charIndex);
      charIndex++;
      setTimeout(typeChar, 45);
    } else {
      const line = document.createElement("div");
      line.className = "terminal-output-line";
      line.textContent = cmd.output;
      output.appendChild(line);

      cmdIndex++;
      charIndex = 0;
      area.textContent = "";

      if (cmdIndex < terminalCommands.length) {
        setTimeout(typeChar, cmd.delay);
      } else {
        setTimeout(() => {
          if (chart) chart.classList.add("visible");
        }, 600);
      }
    }
  }

  setTimeout(typeChar, 600);
}

function initMobileNav() {
  const btn = document.getElementById("menu-btn");
  const nav = document.getElementById("mobile-nav");
  if (!btn || !nav) return;

  btn.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    btn.textContent = open ? "close" : "menu";
  });

  nav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      nav.classList.remove("open");
      btn.textContent = "menu";
    });
  });
}

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const targetId = anchor.getAttribute("href");
      if (targetId === "#") return;
      const target = document.querySelector(targetId);
      if (target) {
        e.preventDefault();
        const navHeight = 64;
        const top = target.getBoundingClientRect().top + window.scrollY - navHeight;
        window.scrollTo({ top, behavior: "smooth" });
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  typeTerminal();
  initMobileNav();
  initSmoothScroll();
});
