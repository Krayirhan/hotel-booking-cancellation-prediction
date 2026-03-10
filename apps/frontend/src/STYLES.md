# Frontend Theme System — Styling Guide

## Overview

The DS Project frontend uses a **two-file CSS architecture** for theming:

| File | Purpose | Activation |
|---|---|---|
| `styles.css` | Classic / Windows-ERP look | Default (no `data-theme` attribute) |
| `modern.css` | Modern light + dark corporate themes | `data-theme="modern-light"` or `data-theme="modern-dark"` on `<html>` |

Both files are imported unconditionally in `main.jsx`. The active theme is determined by the presence (and value) of the `data-theme` attribute on the `<html>` element.

---

## How Theme Switching Works

1. **`useTheme` hook** (`hooks/useTheme.js`) reads theme preference from `localStorage`.
2. **`Layout.jsx`** calls `document.documentElement.setAttribute('data-theme', theme)` whenever the theme changes.
3. CSS specificity takes over: `[data-theme="modern-light"] .some-class` overrides `:root .some-class`.

```
localStorage key: "ds-theme"
Values: "classic" | "modern-light" | "modern-dark"
```

---

## CSS Variable System

All design tokens (colours, spacing, borders) are declared as CSS custom properties (variables). Both files reuse the **same variable names** — changing the theme simply changes the values.

### Key variables (examples)

| Variable | Classic | Modern Light | Modern Dark |
|---|---|---|---|
| `--win-bg` | `#ece9d8` | `#f4f6f9` | `#1e2026` |
| `--win-face` | `#f0f0f0` | `#ffffff` | `#282c34` |
| `--win-dark` | `#1c1c1c` | `#1a2639` | `#f0f4ff` |
| `--accent` | `#0047AB` | `#2563eb` | `#60a5fa` |

---

## File Structure

```
src/
├── styles.css        # Classic theme (1 300+ lines; always loaded)
├── modern.css        # Modern themes (1 600+ lines; always loaded)
└── STYLES.md         # This file
```

### `styles.css` sections (in order)

1. `:root` variable palette
2. Global resets + typography
3. Login / auth pages
4. App shell layout (sidebar, topbar, content area)
5. Component styles (cards, tables, buttons, badges, modals)
6. Page-specific overrides (Models, Guests, Chat, System, Dashboard, Monitoring)

### `modern.css` sections (in order)

1. `[data-theme="modern-light"]` token overrides
2. `[data-theme="modern-dark"]` token overrides
3. Modern component variants (layout, cards, tables, forms)
4. Page-specific modern overrides

---

## Adding a New Component

1. Add base styles in `styles.css` under the appropriate section comment.
2. If the component look differs significantly in modern themes, add overrides in `modern.css` under the matching `[data-theme]` block.
3. Use existing CSS variables (e.g. `var(--win-bg)`) rather than hard-coded colours.

---

## Avoiding Common Mistakes

- **Do not** set colours directly (e.g. `color: #ece9d8`). Always use `var(--variable-name)`.
- **Do not** gate component visibility with the theme — classic/modern differ only in appearance.
- **Do not** add `!important` unless overriding a third-party library rule.
