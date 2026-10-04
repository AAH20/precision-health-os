# UI/UX & Accessibility Contract — Precision Health OS Dashboards

This document defines the binding UI/UX and accessibility contract for the
Precision Health OS web dashboards (`web/dashboard.html`, `web/benchmarks.html`).
It is structured around the `ui-ux-quality` skill rules. The HTML files may not
exist yet at the time of writing; this document specifies the contract they must
satisfy. Any visual claim below is a contract requirement, not a verified
observation of rendered output.

## 1. Design Principles

- **Zero-overlay policy.** Layout uses CSS Grid and Flexbox only. No float-based
  layouts. `position: absolute` is permitted only for tooltips, dropdowns, and
  modals — never for general layout. If two elements could overlap at any viewport
  size, restructure the layout.
- **Mobile-first at 320px.** Design and test for a 320px viewport first, then
  scale up. No horizontal scroll at any size.
- **Containment.** Use `min-height`/`min-width` to prevent collapse; use
  `overflow: hidden` or `overflow: auto` on containers to prevent spill.
- **z-index discipline.** `z-index` only for modals, dropdowns, and tooltips.
  `position: relative` on the parent when a child needs `position: absolute`.

## 2. Responsive Breakpoints

Standard breakpoint set. The layout must be verified at 320, 768, 1024, and
1920 px with no horizontal scroll.

| Breakpoint | Target | What changes |
|---|---|---|
| 320px | Small phones | Base mobile layout. Single-column grids, fluid type via `clamp()`, nav wraps. |
| 640px | Large phones / small tablets | Wider cards, optional 2-column forms. |
| 768px | Tablets | Multi-column card grids begin, nav may stay horizontal. |
| 1024px | Laptops | Full multi-column grids, max-width container engaged. |
| 1280px | Desktops | Comfortable spacing, larger type scale. |
| 1920px | Large monitors | Content capped at max-width; no stretch beyond container. |

- Use relative units (`rem`, `em`, `%`, `vh`, `vw`) for layout; avoid `px` for
  layout dimensions.
- Use `clamp()` for fluid typography, e.g.
  `font-size: clamp(1rem, 2.5vw, 1.5rem)`.
- Use `min()`/`max()` for fluid spacing.
- `body { overflow-x: hidden; }` to prevent horizontal scroll.

## 3. Accessibility Contract

- **Semantic HTML required.** Use `<header>`, `<nav>`, `<main>`, `<section>`,
  `<article>`, `<footer>` as appropriate. Do not use generic `<div>` where a
  semantic element exists.
- **ARIA usage.** Add ARIA labels where semantic HTML is insufficient. Use
  `aria-expanded` for collapsible sections, `aria-hidden="true"` for decorative
  elements, `role="dialog"` and `aria-modal="true"` for modals.
- **WCAG AA contrast.** Minimum 4.5:1 for normal text, 3:1 for large text
  (>=18px regular or >=14px bold). Verify all text/background pairs.
- **Keyboard reachability.** All interactive elements must be reachable and
  operable via Tab/Enter/Space. Test with keyboard only.
- **Focus-visible styles.** Every interactive element must have a visible
  `:focus-visible` outline. Never remove the focus ring without replacing it.

## 4. Motion Policy

- Animate `transform` and `opacity` only. Never animate `width`, `height`, `top`,
  or `left`.
- UI feedback transitions <= 300ms. Page transitions <= 1000ms.
- Easing: `cubic-bezier(0.4, 0, 0.2, 1)`.
- Use `will-change` sparingly — only on elements that actually animate.
- **Reduced-motion fallback.** Always include:
  ```css
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.01ms !important;
      transition-duration: 0.01ms !important;
    }
  }
  ```

## 5. Theming — CSS Custom Properties

All colors, radii, and shadows are defined as `:root` custom properties. Do not
hardcode color values in component styles.

| Token | Purpose |
|---|---|
| `--color-primary` | Primary brand / accent color |
| `--color-secondary` | Secondary accent |
| `--color-bg` | Page background |
| `--color-card` | Card / surface background |
| `--color-text` | Primary text color |
| `--color-muted` | Secondary / muted text |
| `--radius` | Border radius (8px) |
| `--shadow` | Box shadow (`0 2px 8px rgba(0,0,0,0.1)`) |

## 6. Component Patterns

### Cards
```css
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
}
.card {
  background: var(--color-card);
  border-radius: var(--radius);
  padding: 1.5rem;
  box-shadow: var(--shadow);
}
```
Use `gap` for spacing — never margins between grid items.

### Navigation
```css
.nav {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  position: sticky;
  top: 0;
}
```

### Tables
- Include a `<caption>`.
- Use `<thead>` with `<th>` elements carrying `scope="col"` (or `scope="row"`
  for row headers).
- No inline styles.

### Forms
```css
.form {
  display: grid;
  grid-template-columns: 1fr;        /* mobile */
  gap: 1rem;
}
@media (min-width: 768px) {
  .form { grid-template-columns: repeat(2, 1fr); }
}
```
Every `<input>`/`<select>`/`<textarea>` has a `<label for="...">` linked by `id`.

## 7. Pre-Delivery Validation Checklist

Before delivering any UI change, verify all ten items:

- [ ] No horizontal scroll at 320px, 768px, 1024px, 1920px
- [ ] No overlapping elements at any viewport size
- [ ] All interactive elements reachable via keyboard
- [ ] Color contrast meets WCAG AA (4.5:1 normal / 3:1 large)
- [ ] Animations respect `prefers-reduced-motion`
- [ ] Semantic HTML used throughout
- [ ] ARIA labels where needed
- [ ] No `!important` unless absolutely necessary
- [ ] No inline styles unless absolutely necessary
- [ ] CSS is organized and commented

## 8. Viewing the Dashboards Locally

The dashboards are static HTML files with no build step and no server
dependency.

1. Open `web/dashboard.html` directly in any modern browser
   (double-click or `File > Open`).
2. Open `web/benchmarks.html` the same way.

No bundler, transpiler, or local server is required. All CSS is inline in a
`<style>` block; there are no external stylesheet or script fetches.
