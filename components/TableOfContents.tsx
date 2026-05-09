"use client"

/**
 * TableOfContents — sticky in-page navigation for long blog posts.
 *
 * Receives headings extracted server-side (title + slug-id pairs) and uses
 * IntersectionObserver to highlight whichever section is currently on screen.
 * Hidden below xl breakpoint so it never crowds the reading column.
 *
 * The observer uses a top-biased rootMargin so the active item advances as
 * the reader scrolls down rather than only when the heading leaves the
 * viewport entirely.
 */

import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"

export interface TocHeading {
  id: string
  text: string
  level: 2 | 3
}

interface TableOfContentsProps {
  headings: TocHeading[]
}

export default function TableOfContents({ headings }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>("")

  useEffect(() => {
    if (headings.length === 0) return

    const observer = new IntersectionObserver(
      (entries) => {
        // Pick the topmost entry that is intersecting.
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible.length > 0) {
          setActiveId(visible[0].target.id)
        }
      },
      {
        // -96px top offset leaves room for the sticky nav bar (h-16 = 64px)
        // plus a little breathing room. The -55% bottom pushes the "active"
        // trigger into the upper half of the viewport.
        rootMargin: "-96px 0px -55% 0px",
        threshold: 0,
      },
    )

    headings.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [headings])

  if (headings.length < 2) return null

  return (
    <nav aria-label="Table of contents" className="sticky top-24 hidden xl:block">
      <div className="mb-4 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        Contents
      </div>
      <ul className="space-y-0.5 border-l border-border">
        {headings.map(({ id, text, level }) => (
          <li key={id}>
            <a
              href={`#${id}`}
              className={cn(
                "-ml-px block border-l-2 py-1 font-mono text-[11px] leading-snug transition-all duration-200",
                level === 2 ? "pl-4" : "pl-7 text-[10px]",
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
