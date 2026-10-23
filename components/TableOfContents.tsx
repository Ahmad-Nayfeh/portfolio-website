"use client"

/**
 * TableOfContents — sticky in-page navigation for long blog posts.
 *
 * Receives headings extracted server-side (title + slug-id pairs) and uses
 * IntersectionObserver to highlight whichever section is currently on screen.
 * Hidden below the xl breakpoint so it never crowds the reading column.
 */

import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"

export interface TocHeading {
  id: string
  text: string
  level: 2 | 3
}

export default function TableOfContents({ headings }: { headings: TocHeading[] }) {
  const [activeId, setActiveId] = useState<string>("")

  useEffect(() => {
    if (headings.length === 0) return
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible.length > 0) setActiveId(visible[0].target.id)
      },
      // Leave room for the sticky nav bar; trigger in the upper half of the viewport.
      { rootMargin: "-88px 0px -55% 0px", threshold: 0 },
    )
    headings.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })
    return () => observer.disconnect()
  }, [headings])

  if (headings.length < 2) return null

  return (
    <nav aria-label="Table of contents" className="sticky top-24 border-[1.5px] border-foreground bg-card p-4">
      <div className="label mb-3 text-[10px]">Contents</div>
      <ul className="space-y-0.5 border-l border-border">
        {headings.map(({ id, text, level }) => (
          <li key={id}>
            <a
              href={`#${id}`}
              className={cn(
                "-ml-px block border-l-2 py-1 text-[12.5px] leading-snug transition-colors",
                level === 2 ? "pl-3" : "pl-6 text-[12px]",
                activeId === id
                  ? "border-l-accent text-foreground"
                  : "border-l-transparent text-muted-foreground hover:text-foreground",
              )}
            >
              {text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}
