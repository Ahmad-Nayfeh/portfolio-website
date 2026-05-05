import BlogCard from "@/components/BlogCard"
import SectionMasthead from "@/components/editorial/SectionMasthead"
import type { Post } from "@/types"

interface RecentPostsProps {
  posts: Post[]
}

export default function RecentPosts({ posts }: RecentPostsProps) {
  return (
    <section className="py-16 md:py-24">
      <SectionMasthead
        kicker="From the Notebook"
        title="Recent writing"
        description="Notes on engineering, AI papers, and the small surprises that come from building real systems."
        link={{ href: "/blog", label: "Open the notebook" }}
      />

      <div className="grid grid-cols-1 gap-x-6 gap-y-12 md:grid-cols-2 lg:grid-cols-3">
        {posts.map((post, idx) => (
          <BlogCard key={post.slug} post={post} index={idx} />
        ))}
      </div>
    </section>
  )
}
