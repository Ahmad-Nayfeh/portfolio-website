"use client"

import { useState, useCallback } from "react"
import { useRouter, useSearchParams, usePathname } from "next/navigation"
import { Search, X } from "lucide-react"
import BlogCard from "@/components/BlogCard"
import type { Post } from "@/types"
import { cn } from "@/lib/utils"

interface BlogGridProps {
  posts: Post[]
}

export default function BlogGrid({ posts }: BlogGridProps) {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const [searchQuery, setSearchQuery] = useState("")

  const selectedTag = searchParams.get("tag") ?? null

  const setTag = useCallback(
    (tag: string | null) => {
      const params = new URLSearchParams(searchParams.toString())
      if (tag) params.set("tag", tag)
      else params.delete("tag")
      const qs = params.toString()
      const url = qs ? `${pathname}?${qs}` : pathname
      // @ts-expect-error -- typed routes don't model query strings
      router.replace(url, { scroll: false })
    },
    [router, pathname, searchParams],
  )

  const allTags = Array.from(new Set(posts.flatMap((post) => post.tags))).sort()

  const filtered = posts.filter((post) => {
    const q = searchQuery.trim().toLowerCase()
    const matchesSearch =
      q === "" || post.title.toLowerCase().includes(q) || (post.excerpt ?? "").toLowerCase().includes(q)
    const matchesTag = selectedTag ? post.tags.includes(selectedTag) : true
    return matchesSearch && matchesTag
  })

  return (
    <div>
      <div className="mb-8 grid grid-cols-1 gap-6 border-[1.5px] border-foreground bg-card p-4 md:grid-cols-[minmax(240px,320px)_1fr] md:p-5">
        <div>
          <label htmlFor="search-posts" className="label mb-2 block text-[10px]">
            Search
          </label>
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              id="search-posts"
              type="search"
              placeholder="Title or summary"
              className="w-full border border-border bg-background py-2.5 pl-9 pr-3 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-foreground"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>
        {allTags.length > 0 && (
          <div>
            <div className="label mb-2 text-[10px]">Topics</div>
            <div className="flex flex-wrap gap-1.5">
              <TagButton active={selectedTag === null} onClick={() => setTag(null)}>
                All
              </TagButton>
              {allTags.map((tag) => (
                <TagButton key={tag} active={selectedTag === tag} onClick={() => setTag(tag)}>
                  {tag}
                </TagButton>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="mb-2 flex items-center justify-between">
        <span className="label">
          {filtered.length} {filtered.length === 1 ? "entry" : "entries"}
          {selectedTag ? ` · ${selectedTag}` : ""}
        </span>
        {(selectedTag || searchQuery) && (
          <button
            type="button"
            onClick={() => {
              setTag(null)
              setSearchQuery("")
            }}
            className="inline-flex items-center gap-1 font-mono text-[11px] uppercase tracking-[0.14em] text-accent"
          >
            <X size={12} /> Clear
          </button>
        )}
      </div>

      <div className="border-t-[1.5px] border-foreground">
        {filtered.length > 0 ? (
          filtered.map((post) => (
            <BlogCard key={post.slug} post={post} index={posts.findIndex((p) => p.slug === post.slug)} />
          ))
        ) : (
          <p className="py-16 text-center text-muted-foreground">No entries match that filter.</p>
        )}
      </div>
    </div>
  )
}

function TagButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.12em] transition-colors",
        active
          ? "border-foreground bg-foreground text-background"
          : "border-border text-muted-foreground hover:border-foreground hover:text-foreground",
      )}
    >
      {children}
    </button>
  )
}
