// components/MarkdownRenderer.tsx
//
// Markdown renderer for non-blog content: about-story, project descriptions,
// home intro. Blog posts use components/MdxRenderer.tsx instead.
//
// rehype-raw stays — some project .md files contain raw HTML (e.g. <div class=
// "...">) that the project pages depend on. Those files are not migrated to
// MDX in Layer B; if Layer C migrates them, this file can drop rehype-raw.
"use client"

import ReactMarkdown from "react-markdown"
import rehypeRaw from "rehype-raw"
import remarkGfm from "remark-gfm"

interface MarkdownRendererProps {
  content: string
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose dark:prose-invert max-w-none prose-lg">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          a: ({ node, ...props }) => {
            if (props.href && (props.href.startsWith("http") || props.href.startsWith("//"))) {
              return <a {...props} target="_blank" rel="noopener noreferrer" />
            }
            return <a {...props} />
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
