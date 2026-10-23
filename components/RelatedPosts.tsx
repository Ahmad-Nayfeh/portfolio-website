import BlogCard from "@/components/BlogCard"
import type { Post } from "@/types"

export default function RelatedPosts({ posts }: { posts: Post[] }) {
  return (
    <div className="border-t-[1.5px] border-foreground">
      {posts.map((post, i) => (
        <BlogCard key={post.slug} post={post} index={i} />
      ))}
    </div>
  )
}
