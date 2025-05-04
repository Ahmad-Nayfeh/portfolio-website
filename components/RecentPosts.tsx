import Link from "next/link"
import { ArrowRight } from "lucide-react"
import BlogCard from "@/components/BlogCard"
import type { Post } from "@/types"

interface RecentPostsProps {
  posts: Post[]
}

export default function RecentPosts({ posts }: RecentPostsProps) {
  return (
    <section className="py-16">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-bold">Recent Posts</h2>
        <Link href="/blog" className="flex items-center text-muted-foreground hover:text-foreground transition-colors">
          View all
          <ArrowRight className="ml-1 h-4 w-4" />
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {posts.map((post) => (
          <BlogCard key={post.slug} post={post} />
        ))}
      </div>
    </section>
  )
}
