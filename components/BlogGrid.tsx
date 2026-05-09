"use client"

/**
 * BlogGrid — filterable post list with URL-reflected state.
 *
 * Tag selection is written into `?tag=<name>` so filtered views are
 * linkable and survive page refresh. The search query stays client-only
 * (no URL state) because it changes on every keystroke and polluting
 * history with search terms is annoying UX.
 */

import { useState, useCallback } from "react"
import { useRouter, useSearchParams, usePathname } from "next/navigation"
import { Search } from "lucide-react"
import BlogCard from "@/components/BlogCard"
import type { Post } from "@/types"
import { cn } from "@/lib/utils"

interface BlogGridProps {
  posts: Post[]
  /** Pre-selected tag read from the URL by the page server component. */
  initialTag?: string | null
}

export default function BlogGrid({ posts, initialTag = null }: BlogGridProps) {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()

  const [searchQuery, setSearchQuery] = useState("")

  // The active tag is the URL param (controlled) not local state.
  const selectedTag = searchParams.get("tag") ?? null

  const setTag = useCallback(
    (tag: string | null) => {
      const params = new URLSearchParams(searchParams.toString())
      if (tag) {
        params.set("tag", tag)
      } else {
        params.delete("tag")
      }
      // Replace so the back button doesn't cycle through every tag click.
      router.replace(`${pathname}?${params.toString()}`, { scroll: false })
    },
    [router, pathname, searchParams],
  )

  // Extract all unique tags from posts.
  const allTags = Array.from(new Set(posts.flatMap((post) => post.tags))).sort()

  // Filter posts by search query and selected tag.
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
      {/* Filter row */}
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
                onClick={() => setTag(null)}
              >
                All
              </button>
              {all