// components/MarkdownRenderer.tsx
"use client"

import ReactMarkdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"
import rehypeRaw from "rehype-raw" // <--- قم باستيراد المكتبة الجديدة

interface MarkdownRendererProps {
  content: string
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose dark:prose-invert max-w-none prose-lg">
      <ReactMarkdown
        rehypePlugins={[
          rehypeRaw, // <--- أضف المكتبة هنا
          rehypeHighlight
        ]}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}