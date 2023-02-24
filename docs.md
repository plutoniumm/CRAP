# Web Components
## `<pa-per>` Usage
```html
<pa-per
href="storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf"
title="A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"
tags="RL">
</pa-per>
```
```ts
// pa-per.ts Attribute Schema
const schema = {
  href: URL; // may or may not be have http(s)://
  title: string;
  tags?: Array<string> | string; // listed in window.$docsify
  icon?: string | `${string}|${string}`; // fas name or icon-set|icon-name (MUST BE SVG)
}
```

<pa-per href="storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf" title="A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"
tags="RL">
</pa-per>