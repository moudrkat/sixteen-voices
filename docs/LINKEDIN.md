## Post

You know what is beautiful about tiny models? That they are tiny.

Twenty-one million parameters. One attention layer. Sixteen heads. You could literally put this model in your pocket, if it existed physically. I trained 82 LoRA adapters — one per author — on CPU, because if your model fits in a pocket you don't need a data center.

Each adapter modifies the same 16 heads, but differently. When you measure how much each head's weights change per author, no two authors look the same. The most variable head differs between Q and V projections — where the model looks vs what it says are shaped by different heads.

It's a small model trained on children's stories. It won't write like Poe. But it's small enough that you can see everything, and that's the point.

Full write-up + code in comments.

## Image

figures/head_importance.png

## Comments

1. Full article: ARTICLE.md (link to repo)
2. Code: github link