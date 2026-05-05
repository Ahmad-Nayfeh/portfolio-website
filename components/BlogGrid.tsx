"use client"

import { useState } from "react"
import { Search } from "lucide-react"
import BlogCard from "@/components/BlogCard"
import type { Post } from "@/types"
import { cn } from "@/lib/utils"

interface BlogGridProps {
  posts: Post[]
}

export default function BlogGrid({ posts }: BlogGridProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedTag, setSelectedTag] = useState<string | null>(null)

  // Extract all unique tags from posts.
  const allTags = Array.from(new Set(posts.flatMap((post) => post.tags))).sort()

  // Filter posts by search query and selected tag.
  const filteredPosts = posts.filter((post) => {
    const q = searchQuery.toLowerCase()
    const matchesSearch =
      post.title.toLowerCase().includes(q) ||
      (post.excerpt ?? "").toLowerCase().includes(q)
    const matchesTag = selectedTag ? post.tags.includes(selectedTag) : true
    return matchesSearch && matchesTag
  })

  return (
    <div>
      {/* Filter row — borderless on top, single bottom rule, editorial. */}
      <div className="mb-12 flex flex-col gap-5 border-b border-border pb-8 lg:flex-row lg:items-end lg:justify-between">
        <div className="lg:max-w-md lg:flex-1">
          <label
            htmlFor="search-posts"
            className="mb-2 block font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground"
          >
            Search the archive
          </label>
          <div className="relative">
            <Search className="pointer-events-none absolute left-0 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              id="search-posts"
              type="search"
              placeholder="Title or excerpt…"
              className="w-full border-0 border-b border-border bg-transparent py-2 pl-7 font-display text-lg outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-accent"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        {allTags.length > 0 && (
          <div className="lg:max-w-xl lg:flex-1">
            <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              Filter by topic
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-1.5 font-mono text-[11px] uppercase tracking-[0.18em]">
              <button
                type="button"
                className={cn(
                  "transition-colors",
                  selectedTag === null
                    ? "text-accent"
                    : "text-muted-foreground hover:text-foreground",
                )}
                onClick={() => setSelectedTag(null)}
              >
                All
              </button>
              {allTags.map((tag) => (
                <button
                  type="button"
                  key={tag}
                  className={cn(
                    "transition-colors",
                    selectedTag === tag
                      ? "text-accent"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                  onClick={() => setSelectedTag(tag)}
                >
                  #{tag}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-x-6 gap-y-12 md:grid-cols-2 lg:grid-cols-3">
        {filteredPosts.map((post, idx) => (
          <BlogCard key={post.slug} post={post} index={idx} />
        ))}
      </div>

      {filteredPosts.length === 0 && (
        <div className="border-y border-border py-20 text-center">
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            No entries match the current filter.
          </p>
        </div>
      )}
    </div>
  )
}
