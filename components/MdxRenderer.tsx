// components/MdxRenderer.tsx
//
// MDX renderer for blog posts. Server component (App Router).
// Pipeline:
//   - remark-gfm        : GitHub-flavored markdown (tables, task lists, autolinks)
//   - remark-math       : recognize $...$ (inline) and $$...$$ (block) as math
//   - rehype-katex      : render math via KaTeX (CSS imported in app/layout.tsx)
//   - rehype-pretty-code: syntax highlight via Shiki (server-rendered tokens)
//
// rehype-raw is intentionally NOT included: MDX already supports JSX, so raw
// HTML passthrough is unnecessary and would conflict with JSX parsing.

import { MDXRemote } from "next-mdx-remote/rsc"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import rehypePrettyCode from "rehype-pretty-code"
import rehypeSlug from "rehype-slug"
import type { Options as PrettyCodeOptions } from "rehype-pretty-code"

const prettyCodeOptions: PrettyCodeOptions = {
  // Pair of themes so light/dark mode both look intentional.
  // Tailwind's `prose-invert` swaps based on `.dark` ancestor.
  theme: {
    light: "github-light",
    dark: "github-dark",
  },
  keepBackground: false,
  defaultLang: "plaintext",
}

interface MdxRendererProps {
  source: string
}

export default function MdxRenderer({ source }: MdxRendererProps) {
  return (
    <div className="prose dark:prose-invert max-w-none prose-lg">
      <MDXRemote
        source={source}
        options={{
          mdxOptions: {
            remarkPlugins: [remarkGfm, remarkMath],
            rehypePlugins: [
              rehypeSlug,
              rehypeKatex,
              [rehypePrettyCode, prettyCodeOptions],
            ],
          },
        }}
        components={{
          a: (props) => {
            const href = props.href || ""
            if (href.startsWith("http") || href.startsWith("//")) {
              return <a {...props} target="_blank" rel="noopener noreferrer" />
            }
            return <a {...props} />
          },
        }}
      />
    </div>
  )
}
