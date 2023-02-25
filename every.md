## A single list of every paper we ever read

<!-- tags from papers -->
<div class="ƒ p5">
  <span v-for="tag in new Set(papers.flatMap( e=> e.get('tags') ))" class="rpm-5 †c"
  style="border: 1px solid #8884; background: #fff;min-width:80px;"
  :id="'tag-' + tag"
  >{{tag}}</span>
</div>

There are a total of <span>{{papers.length}}</span> papers.

<div v-for="(p,i) in papers.map( Object.fromEntries )" :key="p.href">
  <pa-per :href="p.href" :title="p.title" :tags="`${p.tags},<b>${i+1}</b>`"></pa-per>
</div>