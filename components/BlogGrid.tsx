"use client"

import { useState, useCallback } from "react"
import { useRouter, useSearchParams, usePathname } from "next/navigation"
import { Search, X } from "lucide-react"
import BlogCard from "@/components/BlogCard"
import StaggerGroup from "@/components/ui/StaggerGroup"
import type { Post } from "@/types"
import { cn } from "@/lib/utils"

interface BlogGridProps {
  posts: Post[]
  initialTag?: string | null
}

export default function BlogGrid({ posts, initialTag = null }: BlogGridProps) {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()

  const [searchQuery, setSearchQuery] = useState("")

  const selectedTag = searchParams.get("tag") ?? null

  const setTag = useCallback(
    (tag: string | null) => {
      const params = new URLSearchParams(searchParams.toString())
      if (tag) {
        params.set("tag", tag)
      } else {
        params.delete("tag")
      }
      const url = `${pathname}?${params.toString()}`
      // @ts-expect-error
      router.replace(url, { scroll: false })
    },
    [router, pathname, searchParams],
  )

  const allTags = Array.from(new Set(posts.flatMap((post) => post.tags))).sort()

  const filteredPosts = posts.filter((post) => {
    const q = searchQuery.toLowerCase()
    const matchesSearch =
      q === "" ||
      post.title.toLowerCase().includes(q) ||
      (post.excerpt ?? "").toLowerCase().includes(q)
    const matchesTag = selectedTag ? post.tags.includes(selectedTag) : true
    return matchesSearch && matchesTag
  })

  return (
    <div>
      {/* Filter section */}
      <div className="mb-12 space-y-6">
        {/* Search */}
        <div className="max-w-md">
          <label
            htmlFor="search-posts"
            className="mb-2 block font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground"
          >
            Search
          </label>
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              id="search-posts"
              type="search"
              placeholder="Title or excerpt…"
              className="w-full rounded-lg border bg-card py-2.5 pl-10 pr-4 font-sans text-sm outline-none transition-colors placeholder:text-muted-foreground/50"
              style={{
                borderColor: "hsl(var(--border))",
                color: "hsl(var(--foreground))",
              }}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        {/* Filter pills */}
        {allTags.length > 0 && (
          <div>
            <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              Topics
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setTag(null)}
                className={cn(
                  "rounded-full border px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.15em] transition-all duration-200",
                  selectedTag === null
                    ? "border-transparent"
                    : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground",
                )}
                style={
                  selectedTag === null
                    ? { backgroundColor: "hsl(var(--accent))", color: "hsl(var(--primary-foreground))" }
                    : {}
                }
              >
                All
              </button>
              {allTags.map((tag) => (
                <button
                  type="button"
                  key={tag}
                  onClick={() => setTag(tag)}
                  className={cn(
                    "rounded-full border px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.15em] transition-all duration-200",
                    selectedTag === tag
                      ? "border-transparent"
                      : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground",
                  )}
                  style={
                    selectedTag === tag
                      ? { backgroundColor: "hsl(var(--accent))", color: "hsl(var(--primary-foreground))" }
                      : {}
                  }
                >
                  {tag}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Active filter indicator */}
        {selectedTag && (
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-muted-foreground">Filtered:</span>
            <button
              type="button"
              onClick={() => setTag(null)}
              className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 font-mono text-[10px] uppercase tracking-[0.15em] transition-colors"
              style={{
                backgroundColor: "hsl(var(--accent) / 0.15)",
                color: "hsl(var(--accent))",
                border: "1px solid hsl(var(--accent) / 0.3)",
              }}
            >
              {selectedTag}
              <X size={10} />
            </button>
          </div>
        )}
      </div>

      {/* Card grid with staggered entrance */}
      {filteredPosts.length > 0 ? (
        <StaggerGroup
          staggerDelay={100}
          direction="up"
          distance={16}
          className="grid grid-cols-1 gap-x-6 gap-y-12 md:grid-cols-2 lg:grid-cols-3"
        >
          {filteredPosts.map((post) => (
            <BlogCard key={post.slug} post={post} />
          ))}
        </StaggerGroup>
      ) : (
        <div className="border-y py-20 text-center" style={{ borderColor: "hsl(var(--border))" }}>
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            No entries match the current filter.
          </p>
        </div>
      )}
    </div>
  )
}
