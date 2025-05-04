// components/MarkdownRenderer.tsx
"use client"

import ReactMarkdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"

interface MarkdownRendererProps {
  content: string
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  // Apply the className to a wrapping div
  return (
    <div className="prose dark:prose-invert max-w-none prose-lg">
      <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
        {content}
      </ReactMarkdown>
    </div>
  )
}