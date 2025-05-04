import type { Metadata } from "next"
import BlogGrid from "@/components/BlogGrid"
import { getAllPosts } from "@/lib/content"

export const metadata: Metadata = {
  title: "Blog | Your Portfolio",
  description: "Articles, tutorials, and thoughts on web development and design",
}

export default async function BlogPage() {
  const posts = await getAllPosts()

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-8">Blog</h1>
      <p className="text-lg mb-8 text-muted-foreground">
        Articles, tutorials, and thoughts on web development and design.
      </p>
      <BlogGrid posts={posts} />
    </div>
  )
}
