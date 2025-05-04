import BlogCard from "@/components/BlogCard"
import type { Post } from "@/types"

interface RelatedPostsProps {
  posts: Post[]
}

export default function RelatedPosts({ posts }: RelatedPostsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {posts.map((post) => (
        <BlogCard key={post.slug} post={post} />
      ))}
    </div>
  )
}
