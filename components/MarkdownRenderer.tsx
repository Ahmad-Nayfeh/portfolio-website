// components/MarkdownRenderer.tsx
"use client"

import ReactMarkdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"
import rehypeRaw from "rehype-raw" // <--- تأكدنا من وجوده
import remarkGfm from 'remark-gfm'   // <--- إضافة مكتبة لدعم الجداول بشكل أفضل

interface MarkdownRendererProps {
  content: string
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose dark:prose-invert max-w-none prose-lg">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]} // <-- إضافة لدعم الجداول
        rehypePlugins={[
          rehypeRaw,               // <-- هذا هو الأهم، لعرض الـ HTML
          rehypeHighlight
        ]}
        // المكونات التالية تضمن أن الروابط الخارجية تفتح في تبويب جديد
        components={{
          a: ({ node, ...props }) => {
            if (props.href && (props.href.startsWith('http') || props.href.startsWith('//'))) {
              return <a {...props} target="_blank" rel="noopener noreferrer" />;
            }
            return <a {...props} />;
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}