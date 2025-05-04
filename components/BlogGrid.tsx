"use client"

import { useState } from "react"
import { Search } from "lucide-react"
import BlogCard from "@/components/BlogCard"
import type { Post } from "@/types"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"

interface BlogGridProps {
  posts: Post[]
}

export default function BlogGrid({ posts }: BlogGridProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedTag, setSelectedTag] = useState<string | null>(null)

  // Extract all unique tags from posts
  const allTags = Array.from(new Set(posts.flatMap((post) => post.tags))).sort()

  // Filter posts by search query and selected tag
  const filteredPosts = posts.filter((post) => {
    const matchesSearch =
      post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.excerpt.toLowerCase().includes(searchQuery.toLowerCase())

    const matchesTag = selectedTag ? post.tags.includes(selectedTag) : true

    return matchesSearch && matchesTag
  })

  return (
    <div>
      <div className="mb-8 space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search posts..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="flex flex-wrap gap-2">
          <Badge
            variant={selectedTag === null ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setSelectedTag(null)}
          >
            All
          </Badge>
          {allTags.map((tag) => (
            <Badge
              key={tag}
              variant={selectedTag === tag ? "default" : "outline"}
              className="cursor-pointer"
              onClick={() => setSelectedTag(tag)}
            >
              {tag}
            </Badge>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredPosts.map((post) => (
          <BlogCard key={post.slug} post={post} />
        ))}
      </div>

      {filteredPosts.length === 0 && (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No posts found matching your criteria.</p>
        </div>
      )}
    </div>
  )
}
