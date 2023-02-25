# Web Components
## `<pa-per>` Usage
```jsx
<pa-per
href="storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf"
title="A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"
tags="RL">
</pa-per>
```
```ts
// pa-per.ts Attribute Schema
interface Paper = {
  href: URL; // may or may not be have http(s)://
  title: string;
  tags?: Array<string> | string; // listed in window.$docsify
  icon?: string | `${string}|${string}`; // fas name or icon-set|icon-name (MUST BE SVG)
}
```

<pa-per href="storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf" title="A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"
tags="RL">
</pa-per>

## `<i-c>` Usage
```jsx
<i-c
src="fas:user"
// EITHER
size="1.75" // multiplies by 16px
// OR
height="16px"
width="16px"

alt="User"
style="filter:invert(86%);">
</i-c>
```
```ts
// i-c.ts Attribute Schema
interface IC = {
  src: string; // in UrsusAPI format (for SVGs only)
  size?: number;
  height?: string;
  width?: string;
  alt?: string;
  style?: string;
}
```

<i-c src="fas:google" height="16px" width="16px" alt="Google" style="filter:invert(86%);"></i-c>